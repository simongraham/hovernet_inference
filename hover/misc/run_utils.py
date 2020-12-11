import cv2
import sys
import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import itertools

from hover.postproc.process_utils import process


def post_proc_para_wrapper(pred_map_mmap_path, tile_info, func_kwargs):
    """Post processing parallel wrapper. Loads a tile from the memory map
    and applies post processing.

    Args:
        pred_map_mmap_path: path to memory map 
        tile_info: coordinate information of post processing tiles
        func_kwargs: function keyword arguments

    Return:
        post processed tile and tile coordinate info

    """
    idx, tile_tl, tile_br = tile_info
    wsi_pred_map_ptr = np.load(pred_map_mmap_path, mmap_mode="r")
    tile_pred_map = wsi_pred_map_ptr[tile_tl[0] : tile_br[0], tile_tl[1] : tile_br[1]]
    tile_pred_map = np.array(tile_pred_map)  # from mmap to ram
    return process(tile_pred_map, **func_kwargs), tile_info


def remove_inst(inst_map, remove_id_list):
    """Remove a nuclear instance from the prediction

    Args:
        inst_map: 2D nuclear instance map
        remove_id_list: list of ids to remove
    
    Return:
        inst_map: refined 2D instance map

    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


def assemble_and_flush(wsi_pred_map_mmap_path, tile_info, factor_40_base, patch_output_list):
    """Assemble results and flush

    Args:
        wsi_pred_map_mmap_path: path to wsi memory map
        tile_info: coordinate information for inference tiles
        factor_40_base: scale factor between wsi at 40x and scanned wsi
        patch_output_list: list of processed output patches

    """
    # write to newly created holder for this wsi
    wsi_pred_map_ptr = np.load(wsi_pred_map_mmap_path, mmap_mode="r+")
    tile_pred_map = wsi_pred_map_ptr[
        tile_info[1][0][0]*factor_40_base: tile_info[1][1][0]*factor_40_base,
        tile_info[1][0][1]*factor_40_base: tile_info[1][1][1]*factor_40_base,
    ]
    if patch_output_list is None:
        tile_pred_map[:] = 0  # zero flush when there are no results
        return

    for pinfo in patch_output_list:
        pcoord, pdata = pinfo
        pdata = np.squeeze(pdata)
        pcoord = np.squeeze(pcoord)[:2]
        tile_pred_map[
            pcoord[0] : pcoord[0] + pdata.shape[0],
            pcoord[1] : pcoord[1] + pdata.shape[1],
        ] = pdata
    return


def get_patch_top_left_info(img_shape, input_size, output_size):
    """Get the top left corner coordinate information

    Args:
        img_shape: input image shape
        input_size: shape of input tiles/patches considered within the image 
        output_size: output shape of each tile/patch
    
    Return:
        input_tl: top left coordinates of the input
        output_tl: top left coordinates of the output

    """
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(
        in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32
    )
    output_tl_x_list = np.arange(
        in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32
    )
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack(
        [output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1
    )
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl


#### all must be np.array
def get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    """Get the coordinate information of tiles used in post processing

    Args:
        img_shape: entire input image shape
        tile_shape: shape of tile used in post processing
        ambiguous_shape: number of pixels from the boundary likely to contain 'border nuclei'
    
    Return:
        tile_grid, tile_boundary, tile_cross: coordinates of tiles for post processing

    """
    # * get normal tiling set
    tile_grid_top_left, _ = get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:, 1])
    tile_grid_y = np.unique(tile_grid_top_left[:, 0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left = np.meshgrid(
        tile_grid_y, tile_grid_x[1:] - ambiguous_size
    )
    tile_boundary_x_bot_right = np.meshgrid(
        tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size
    )
    tile_boundary_x_top_left = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack(
        [tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1
    )
    #
    tile_boundary_y_top_left = np.meshgrid(
        tile_grid_y[1:] - ambiguous_size, tile_grid_x
    )
    tile_boundary_y_bot_right = np.meshgrid(
        tile_grid_y[1:] + ambiguous_size, tile_grid_x + tile_shape[1]
    )
    tile_boundary_y_top_left = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack(
        [tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1
    )
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    tile_cross_top_left = np.meshgrid(
        tile_grid_y[1:] - 2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size
    )
    tile_cross_bot_right = np.meshgrid(
        tile_grid_y[1:] + 2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size
    )
    tile_cross_top_left = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross


def get_tile_patch_info(
    img_shape, tile_input_shape, patch_input_shape, patch_output_shape):
    """Get the coordinate information for tiles and patches during inference

    Args:
        img_shape: shape of input WSI
        tile_input_shape: shape of tiles used during inference
        patch_input_shape: shape of input patches
        patch_output_shape: shape of output patches
    
    Return:
        tile_info_list: coordinates of tiles used during inference
        patch_info_list: coordinates of patches

    """
    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    tile_output_shape = tile_input_shape - patch_diff_shape
    tile_output_shape = round_to_multiple(tile_output_shape, patch_output_shape).astype(
        np.int64
    )
    tile_input_shape = (tile_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = get_patch_top_left_info(
        img_shape, patch_input_shape, patch_output_shape
    )
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape
    patch_info_list = np.stack(
        [
            np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
            np.stack([patch_output_tl_list, patch_output_br_list], axis=1),
        ],
        axis=1,
    )

    tile_input_tl_list, _ = get_patch_top_left_info(
        img_shape, tile_input_shape, tile_output_shape
    )
    tile_input_br_list = tile_input_tl_list + tile_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(tile_input_br_list[:, 0] > img_shape[0])[0]
    x_sel = np.nonzero(tile_input_br_list[:, 1] > img_shape[1])[0]
    tile_input_br_list[y_sel, 0] = (
        img_shape[0] - patch_diff_shape[0]
    ) - tile_input_tl_list[y_sel, 0]
    tile_input_br_list[x_sel, 1] = (
        img_shape[1] - patch_diff_shape[1]
    ) - tile_input_tl_list[x_sel, 1]
    tile_input_br_list[y_sel, 0] = round_to_multiple(
        tile_input_br_list[y_sel, 0], patch_output_shape[0]
    )
    tile_input_br_list[x_sel, 1] = round_to_multiple(
        tile_input_br_list[x_sel, 1], patch_output_shape[1]
    )
    tile_input_br_list[y_sel, 0] += tile_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    tile_input_br_list[x_sel, 1] += tile_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    tile_output_tl_list = tile_input_tl_list + patch_diff_shape // 2
    tile_output_br_list = tile_input_br_list - patch_diff_shape // 2  # may off pixels
    tile_info_list = np.stack(
        [
            np.stack([tile_input_tl_list, tile_input_br_list], axis=1),
            np.stack([tile_output_tl_list, tile_output_br_list], axis=1),
        ],
        axis=1,
    )

    return tile_info_list, patch_info_list
