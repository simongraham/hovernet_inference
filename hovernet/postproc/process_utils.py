import cv2
import numpy as np
from scipy.ndimage.morphology import (binary_dilation, binary_erosion, binary_closing)

import skimage
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.filters import rank, threshold_otsu
from scipy import ndimage

from hovernet.config import Config
from hovernet.postproc import hover
from skimage.measure import regionprops

from hovernet.misc.utils import bounding_box

####
def process_instance(pred_map, type_classification, nr_types, remap_label=False, output_dtype='uint16'):
    # Post processing
    cfg = Config()

    if type_classification:
        pred_inst = pred_map[..., nr_types:]
        pred_type = pred_map[..., :nr_types]

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type, axis=-1)
        pred_type = np.squeeze(pred_type)

    else:
        pred_inst = pred_map

    pred_inst = hover.proc_np_hv(pred_inst,
                                            marker_mode=2,
                                            energy_mode=2, rgb=None)
    
    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)
    
    if type_classification:
        pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])               
        #### * Get class of each instance id, stored at index id-1
        pred_id_list = list(np.unique(pred_inst))[1:] # exclude background ID
        pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
        for idx, inst_id in enumerate(pred_id_list):
            inst_tmp = pred_inst == inst_id
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0: # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]
            pred_type_out += (inst_tmp * inst_type)
        pred_type_out = pred_type_out.astype(output_dtype)
    else:
        pred_type_out = None

    pred_inst = pred_inst.astype(output_dtype)
    
    return pred_inst, pred_type_out
####

def process_instance_wsi(pred_map, type_classification, nr_types, patch_coords, remap_label=False, offset=0,
                         scan_resolution=0.25, get_summary=True, output_dtype='uint16'):
    # Post processing
    cfg = Config()
    mask_list_out = []
    type_list_out = []
    cent_list_out = []

    if type_classification:
        pred_inst = pred_map[..., nr_types:]
        pred_type_ = pred_map[..., :nr_types]
        pred_type_ = np.squeeze(pred_type_)

        pred_inst = np.squeeze(pred_inst)
        pred_type = np.argmax(pred_type_, axis=-1)
        pred_type = np.squeeze(pred_type)

    else:
        pred_inst = pred_map

    pred_inst = hover.proc_np_hv(pred_inst,
                                          marker_mode=2,
                                          energy_mode=2, rgb=None)

    # remap label is very slow - only uncomment if necessary to map labels in order
    if remap_label:
        pred_inst = remap_label(pred_inst, by_size=True)

    
    pred_type_out = np.zeros([pred_type.shape[0], pred_type.shape[1]])
    #### * Get class of each instance id, stored at index id-1
    pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
    pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)

    for idx, inst_id in enumerate(pred_id_list):
        inst_tmp = pred_inst == inst_id

        inst_tmp = inst_tmp.astype('uint8')

        # get the cropped mask
        [rmin, rmax, cmin, cmax] = bounding_box(inst_tmp)
        cropped_inst_ = inst_tmp[rmin:rmax, cmin:cmax]
        cropped_inst = np.zeros([cropped_inst_.shape[0] + 2, cropped_inst_.shape[1] + 2])
        cropped_inst[1:cropped_inst.shape[0] - 1, 1:cropped_inst.shape[1] - 1] = cropped_inst_

        if scan_resolution > 0.35:  # it means image is scanned at 20X
            cropped_inst = cv2.resize(cropped_inst, dsize=(int(cropped_inst.shape[1]/2), int(cropped_inst.shape[0]/2)), interpolation=cv2.INTER_NEAREST)

        cropped_inst = cropped_inst.astype('bool')
        mask_list_out.append(cropped_inst)

        # get the centroid
        regions = regionprops(inst_tmp)
        centroid = np.array(regions[0].centroid)
        centroid += offset  # offset due to the difference between image and mask size
        if scan_resolution > 0.35:  # it means image is scanned at 20X
            centroid /= 2

        centroid += patch_coords
        cent_list_out.append(centroid)
        summary_prob_tmp = []

        if type_classification:
            # get the type
            inst_type = pred_type[pred_inst == inst_id]
            type_list, type_pixels = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
            inst_type = type_list[0][0]
            if inst_type == 0:  # ! pick the 2nd most dominant if exist
                if len(type_list) > 1:
                    inst_type = type_list[1][0]

            type_list_out.append(inst_type)

    return mask_list_out, type_list_out, cent_list_out

####

def img_min_axis(img):
    try:
        return min(img.shape[:2])
    except AttributeError:
        return min(img.size)
####

def stain_entropy_otsu(img, proc_scale):
    '''
    Description
    '''

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
####

def morphology(mask, proc_scale):
    '''
    Description
    '''

    mask_scale = img_min_axis(mask)
    # Join together large groups of small components ('salt')
    radius = int(8 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    # Remove thin structures
    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_erosion(mask, selem)

    # Remove small disconnected objects
    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Close up small holes ('pepper')
    mask = binary_closing(mask, selem)

    mask = remove_small_objects(
        mask,
        min_size=int(120 * proc_scale)**2,
        connectivity=1,
    )

    radius = int(16 * proc_scale)
    selem = disk(radius)
    mask = binary_dilation(mask, selem)

    mask = remove_small_holes(
        mask,
        area_threshold=int(40 * proc_scale)**2,
        connectivity=1,
    )

    # Fill holes in mask
    mask = ndimage.binary_fill_holes(mask)

    return mask
####

def get_tissue_mask(img, proc_scale=0.5):
    '''
    Description
    '''
    img_copy = img.copy()
    if proc_scale != 1.0:
        img_resize = cv2.resize(img_copy, None, fx=proc_scale, fy=proc_scale)
    else:
        img_resize = img_copy

    mask = stain_entropy_otsu(img_resize, proc_scale)
    mask = morphology(mask, proc_scale)
    mask = mask.astype('uint8')

    if proc_scale != 1.0:
        mask = cv2.resize(
            mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask
####

def remap_label(pred, by_size=False):
    """
    Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger instances has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1    
    return new_pred
#####
