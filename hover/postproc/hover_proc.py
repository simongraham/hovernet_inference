
import cv2
import numpy as np
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects, watershed



def proc_np_hv(pred, return_coords=False):
    """
    Process Nuclei Prediction with XY Coordinate Map

    Args:
        pred:           prediction output, assuming 
                        channel 0 contain probability map of nuclei
                        channel 1 containing the regressed X-map
                        channel 2 containing the regressed Y-map
        return_coords: return coordinates of extracted instances
    """

    blb_raw = pred[...,0]
    h_dir_raw = pred[...,1]
    v_dir_raw = pred[...,2]

    # Processing 
    blb = np.copy(blb_raw)
    blb[blb >= 0.5] = 1
    blb[blb <  0.5] = 0

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1 # background is 0 already
    #####

    h_dir = cv2.normalize(h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    h_dir_raw = None  # clear variable
    v_dir_raw = None  # clear variable

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)
    h_dir = None  # clear variable
    v_dir = None  # clear variable

    sobelh = 1 - (cv2.normalize(sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    sobelv = 1 - (cv2.normalize(sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

    overall = np.maximum(sobelh, sobelv)
    sobelh = None  # clear variable
    sobelv = None  # clear variable
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    # nuclei values form peaks so inverse to get basins
    dist = -cv2.GaussianBlur(dist,(3, 3),0)

    overall[overall >= 0.5] = 1
    overall[overall <  0.5] = 0
    marker = blb - overall
    overall = None # clear variable
    marker[marker < 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
 
    pred_inst = watershed(dist, marker, mask=blb, watershed_line=False)
    if return_coords:
        label_idx = np.unique(pred_inst)
        coords = measurements.center_of_mass(blb, pred_inst, label_idx[1:])
        return pred_inst, coords
    else:
        return pred_inst

