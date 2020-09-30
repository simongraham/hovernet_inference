import glob
import os
import shutil
import requests
import cv2
import numpy as np

import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.filters import rank, threshold_otsu
from skimage.transform import resize
from scipy import ndimage

from scipy.ndimage.morphology import (
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_closing,
)


def get_bounding_box(img):  
    """Get the coordinates of the bounding box of a binary input

    Args:
        img (ndarray): binary 2D input
    
    Return:
        row and column minimum and maximum coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def rm_n_mkdir(dir_path):
    """Remove and then make a directory

    Args:
        dir_path: path where to create directory

    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def type_colour(model_name):
    """Generate RGB colour for overlay based on class id

    Args:
        model_name: either pannuke or monusac
    
    Return: 
        colour_dict: dictionary showing RGB values for each class value

    """
    if model_name == 'pannuke':
        colour_dict = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0), 5: (255, 165, 0)
        }
    elif model_name == 'monusac':
        colour_dict = {
            0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0)
        }

    return colour_dict
    

def visualize_instances(input_image, inst_dict, model_name, line_thickness=2):
    """Generate overlay of results on top of original image

    Args:
        input_image (ndarray): original input image
        inst_dict (dict): dictionary of instance level results
        model_name: either pannuke or monusac - determines overlay colours
        line_thickness: thickness of line used in overlay
    
    Return:
        overlay (ndarray): generated overlay

    """
    overlay = np.copy((input_image).astype(np.uint8))

    colour_dict = type_colour(model_name) # get dictionary of RGB colours for each class

    for idx, [inst_id, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colour is not None:
            inst_colour = colour_dict[inst_info["type"]]
        else:
            inst_colour = (255, 255, 0)
        cv2.drawContours(overlay, [inst_contour], -1, inst_colour, line_thickness)

    return overlay


def stain_entropy_otsu(img):
    """Generate tissue mask using otsu thresholding and entropy calculation

    Args:
        img (ndarray): input image
    
    Return:
        mask (ndarray): binary mask

    """
    img_copy = img.copy()
    hed = skimage.color.rgb2hed(img_copy)  # convert colour space
    hed = (hed * 255).astype(np.uint8)
    h = hed[:, :, 0]
    e = hed[:, :, 1]
    d = hed[:, :, 2]
    selem = disk(4)  # structuring element
    # calculate entropy for each colour channel
    h_entropy = rank.entropy(h, selem)
    e_entropy = rank.entropy(e, selem)
    d_entropy = rank.entropy(d, selem)
    entropy = np.sum([h_entropy, e_entropy], axis=0) - d_entropy
    # otsu threshold
    threshold_global_otsu = threshold_otsu(entropy)
    mask = entropy > threshold_global_otsu

    return mask


def morphology(mask):
    """Apply morphological operation to refine tissue mask

    Args:
        mask (ndarray): 2D binary mask
    
    Return:
        mask (ndarray): refined 2D binary mask

    """
    # Join together large groups of small components ('salt')
    mask = binary_dilation(mask, disk(int(4)))

    # Remove thin structures
    mask = binary_erosion(mask, disk(int(8)))

    # Remove small disconnected objects
    mask = remove_small_holes(mask, area_threshold=int(20) ** 2, connectivity=1,)

    # Close up small holes ('pepper')
    mask = binary_closing(mask,  disk(int(8)))

    mask = remove_small_objects(mask, min_size=int(60) ** 2, connectivity=1,)
    mask = binary_dilation(mask, disk(int(8)))
    mask = remove_small_holes(mask, area_threshold=int(20) ** 2, connectivity=1,)

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask


def get_tissue_mask(img):
    """Get tissue mask of an input image

    Args:
        img (ndarray): input image 
    
    Return:
        mask (ndarray): output tissue mask

    """
    mask = stain_entropy_otsu(img)
    mask = morphology(mask)
    mask = mask.astype("uint8")

    return mask
