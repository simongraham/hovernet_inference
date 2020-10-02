"""run.py

Usage:
  run.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--input_dir=<path>] [--output_dir=<path>] \
      [--cache_dir=<path>] [--batch_size=<n>] [--inf_tile_shape=<n>] [--proc_tile_shape=<n>] \
      [--postproc_workers=<n>] [--return_probs]
  run.py (-h | --help)
  run.py --version

Options:
  -h --help                  Show this string.
  --version                  Show version.
  --gpu=<id>                 GPU list. [default: 1]
  --mode=<mode>              Inference mode. `tile` or `wsi`. [default: wsi]
  --model=<path>             Path to model. Use either `pannuke.npz` or `monusac.npz` [default: pannuke.npz]
  --input_dir=<path>         Directory containing input images/WSIs. [default: wsi_input]
  --output_dir=<path>        Directory where the output will be saved. [default: wsi_output/]
  --cache_dir=<path>         Cache directory for saving temporary output. [default: cache/]
  --batch_size=<n>           Batch size. [default: 25]
  --inf_tile_shape=<n>       Size of tiles for inference (assumes square shape). [default: 10000]
  --proc_tile_shape=<n>      Size of tiles for post processing (assumes square shape). [default: 2048]
  --postproc_workers=<n>     Number of workers for post processing. [default: 0]
  --return_probs             Whether to return the class probabilities for each nucleus

"""


from docopt import docopt
import glob
import math
import os
import sys
import json
import importlib
from collections import deque
from multiprocessing import Pool

import cv2
import numpy as np
import tqdm

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from tensorpack import logger

logger._getlogger().disabled = True  # disable logging of network info

from hover.postproc.process_utils import process
from hover.misc.wsi_handler import get_wsi_handler
from hover.misc.utils import rm_n_mkdir, visualize_instances, get_tissue_mask
from hover.misc.run_utils import (
    remove_inst,
    assemble_and_flush,
    get_tile_info,
    get_tile_patch_info,
    post_proc_para_wrapper
)

import time


class InferTile(object):
    """Tile inference class    

    Attributes:
        nr_types: number of classes(including BG) for nuclear type classification
        patch_input_shape: input dimensions of patch [h, w]
        patch_output_shape: output dimensions of patch [h, w]
        input_norm (bool): whether to normalise the input between 0 and 1
        model_name: which dataset the model was trained on- either `pannuke` or `monusac`
        model_path: path to the npz checkpoint file
        input_dir: input directory containing WSIs
        output_dir: output directory where files will be saved
        batch_size (int): batch size to use during inference
        input_tensor_names: defines what it the input in the computational graph
        output_tensor_names: defines what it the output in the computational graph

    """
    def __init__(self,):
        self.nr_types = None  
        self.patch_input_shape = [256, 256]
        self.patch_output_shape = [164, 164]
        self.input_norm = True  

        self.input_tensor_names = ["images"]
        self.output_tensor_names = ["predmap-coded"]
    
    def _parse_args(self, args):
        """Parse CLI arguments

        Args:
            args: command line interface arguments

        """
        self.model_path = args["--model"]
        self.input_dir = args["--input_dir"]
        self.output_dir = args["--output_dir"]
        self.batch_size = int(args["--batch_size"])
        self.return_probs = args['--return_probs']

        # get the model name from the checkpoint
        model_name = os.path.basename(self.model_path)
        self.model_name = model_name.split('.')[0]
        if self.model_name == 'pannuke':
            self.nr_types = 6
        elif self.model_name == 'monusac':
            self.nr_types = 5

    def get_model(self):
        """Get the model architecture"""
        model_constructor = importlib.import_module("hover.model.graph")
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor  # NOTE return alias, not object

    def __gen_prediction(self, tile, predictor):
        """Using 'predictor' to generate the prediction of input tile

        Args:
            tile: input image to be segmented. It will be split into patches
                  to run the prediction upon before being assembled back
            predictor: A predictor built from a given config.

        Return:
            pred_map: merged probability map

        """
        step_size = self.patch_output_shape
        msk_size = self.patch_output_shape
        win_size = self.patch_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        im_h = tile.shape[0]
        im_w = tile.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        tile = np.lib.pad(tile, ((padt, padb), (padl, padr), (0, 0)), "reflect")

        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range(0, last_w, step_size[1]):
                win = tile[row : row + win_size[0], col : col + win_size[1]]
                sub_patches.append(win)

        pred_list = deque()
        while len(sub_patches) > self.batch_size:
            mini_batch = sub_patches[: self.batch_size]
            sub_patches = sub_patches[self.batch_size :]
            batch_output = predictor(mini_batch)[0]
            batch_output = np.split(batch_output, self.batch_size, axis=0)
            pred_list.extend(batch_output)
        if len(sub_patches) != 0:
            batch_output = predictor(sub_patches)[0]
            batch_output = np.split(batch_output, len(sub_patches), axis=0)
            pred_list.extend(batch_output)

        output_patch_shape = np.squeeze(pred_list[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        # Assemble back into full image
        pred_map = np.squeeze(np.array(pred_list))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = (
            np.transpose(pred_map, [0, 2, 1, 3, 4])
            if ch != 1
            else np.transpose(pred_map, [0, 2, 1, 3])
        )
        pred_map = np.reshape(
            pred_map,
            (
                pred_map.shape[0] * pred_map.shape[1],
                pred_map.shape[2] * pred_map.shape[3],
                ch,
            ),
        )
        pred_map = np.squeeze(pred_map[:im_h, :im_w])  # just crop back to original size

        return pred_map

    def load_model(self):
        """Loads the model and checkpoints"""
        print("Loading Model...")
        model_path = self.model_path
        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model=model_constructor(
                self.nr_types, self.patch_input_shape, self.patch_output_shape, self.input_norm
            ),
            session_init=get_model_loader(model_path),
            input_names=self.input_tensor_names,
            output_names=self.output_tensor_names,
        )
        self.predictor = OfflinePredictor(pred_config)

    def process_all_files(self):
        """Process all image files within a directory. The function will:

        1) Load the image
        2) Extract patches the entire image
        3) Run inference
        4) Return instance prediction, overlay and JSON files

        """
        save_dir = self.output_dir
        file_list = glob.glob("%s/*" % self.input_dir)
        file_list.sort()  # ensure same order

        rm_n_mkdir(save_dir)

        pbar = tqdm.tqdm(desc='Processing Images', leave=True,
                         total=len(file_list),
                         ncols=80, ascii=True, position=0)

        for filename in file_list:
            filename = os.path.basename(filename)
            basename = os.path.splitext(filename)[0]

            rm_n_mkdir(save_dir + '/' + basename)

            ###
            img = cv2.imread(self.input_dir + "/" + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ###
            pred_map = self.__gen_prediction(img, self.predictor)

            pred_inst, pred_info = process(
                pred_map, nr_types=self.nr_types, return_dict=True, return_probs=self.return_probs)

            overlaid_output = visualize_instances(img, pred_info, self.model_name)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

            cv2.imwrite("%s/%s/overlay.png" % (save_dir, basename), overlaid_output)
            np.save("%s/%s/instances.npy" % (save_dir, basename), pred_inst)

            # save result info as json file
            json_dict = {}
            for inst_id, inst_info in pred_info.items():
                new_inst_info = {}
                for info_name, info_value in inst_info.items():
                    # convert to JSON
                    if isinstance(info_value, np.ndarray):
                        info_value = info_value.tolist()
                    new_inst_info[info_name] = info_value
                json_dict[int(inst_id)] = new_inst_info
            with open("%s/%s/nuclei_dict.json" % (save_dir, basename), "w") as handle:
                json.dump(json_dict, handle)
            
            pbar.update()
        pbar.close()


class InferWSI(object):
    """WSI inference class

    Attributes:
        nr_types: number of classes (including BG) for nuclear type classification
        patch_input_shape: input dimensions of patch [h,w]
        patch_output_shape: output dimensions of patch [h,w]
        input_norm (bool): whether to normalise the input between 0 and 1
        model_name: which dataset the model was trained on- either `pannuke` or `monusac`
        wsi_proc_lvl: level of wsi pyramid to process (should use level 0)
        ambiguous_size: number of pixels from boundary likely to contain 'border nuclei'
        wsi_inst_info: dictionary of instance-level results
        inf_tile_shape: dimensions of tiles during inference [h,w]
        proc_tile_shape: dimensions of tiles during post processing [h,w]
        model_path: path to model checkpoints
        input_dir: input directory containing WSIs
        output_dir: output directory where files will be saved
        cache_dir: directory where temporary files will be save (ensure around 100GB space)
        return_probs (bool): whether to return the per class probabilities of each nucleus
        batch_size (int): batch size to use during inference
        postproc_workers (int): number of parallel post processing workers
        input_tensor_names: defines what it the input in the computational graph
        output_tensor_names: defines what it the output in the computational graph

    """
    def __init__(self):
        self.nr_types = None
        self.patch_input_shape = [256, 256]
        self.patch_output_shape = [164, 164]
        self.input_norm = True  
        self.wsi_proc_lvl = 0  
        self.ambiguous_size = 128  
        self.wsi_inst_info = {}

        self.input_tensor_names = ["images"]
        self.output_tensor_names = ["predmap-coded"]
    
    def _parse_args(self, args):
        """Parse CLI arguments

        Args:
            args: command line interface arguments
        
        """
        # tile inference shape
        self.inf_tile_shape = [
            int(args["--inf_tile_shape"]),
            int(args["--inf_tile_shape"]),
        ]
        # tile post processing shape
        self.proc_tile_shape = [
            int(args["--proc_tile_shape"]),
            int(args["--proc_tile_shape"]),
        ]

        self.model_path = args["--model"]
        self.input_dir = os.path.abspath(args["--input_dir"])
        self.output_dir = args["--output_dir"]
        # temporarily stores probability map as memory map in cache - ensure enough space
        self.cache_dir = args["--cache_dir"]
        self.return_probs = args["--return_probs"]
        self.batch_size = int(args["--batch_size"])
        self.postproc_workers = int(args["--postproc_workers"])

        # get the model name from the checkpoint
        model_name = os.path.basename(self.model_path)
        self.model_name = model_name.split('.')[0]
        if self.model_name == 'pannuke':
            self.nr_types = 6
        elif self.model_name == 'monusac':
            self.nr_types = 5

    def get_model(self):
        """Get the model architecture"""
        model_constructor = importlib.import_module("hover.model.graph")
        model_constructor = model_constructor.Model_NP_HV
        return model_constructor  # NOTE return alias, not object

    def __run_inference(self, patch_top_left_list, tile_idx, nr_tiles):
        """Get the raw predictions for a set of input patch coordinates

        Args:
            patch_top_left_list: top left coordinates of patches to process
            tile_idx: index of tile that patches have been extracted from
            nr_tiles: total number of tiles in the WSI
        
        Return:
            pred_list: list of patch-level predictions
        
        """
        cache_tile = np.load("%s/cache_tile.npy" % self.cache_dir)
        cache_tile = np.array(cache_tile)
        sub_patches = []
        patches_info = []
        # generating subpatches from orginal
        for patch_coord in patch_top_left_list:
            win = cache_tile[
                patch_coord[0] : patch_coord[0] + self.patch_input_shape_tmp[0],
                patch_coord[1] : patch_coord[1] + self.patch_input_shape_tmp[0],
            ]
            if self.base_mag_ds > 1:
                # cv.INTER_LINEAR is good for zooming
                win = cv2.resize(
                    win, (self.patch_input_shape[1], self.patch_input_shape[0]), cv2.INTER_LINEAR)
                patch_coord *= self.base_mag_ds
            sub_patches.append(win)
            patches_info.append(patch_coord)

        pred_list = deque()
        batch_count = len(sub_patches)
        nr_proc = 0 # used for logging number of processed batches
        while len(sub_patches) > self.batch_size:
            mini_batch = sub_patches[: self.batch_size]
            sub_patches = sub_patches[self.batch_size:]
            batch_info = patches_info[: self.batch_size]
            patches_info = patches_info[self.batch_size:]
            batch_output = self.predictor(mini_batch)[0]
            batch_output = np.split(batch_output, self.batch_size, axis=0)
            batch_info = np.split(np.array(batch_info), self.batch_size, axis=0)
            batch_output_combined = list(
                zip(batch_info, batch_output))
            pred_list.extend(batch_output_combined)
            ###
            nr_proc += self.batch_size
            sys.stdout.write("\rProcessing Batch (%d/%d) of Tile (%d/%d)" %
                             (nr_proc, batch_count, tile_idx+1, nr_tiles))
            sys.stdout.flush()
        if len(sub_patches) != 0:
            batch_output = self.predictor(sub_patches)[0]
            batch_output = np.split(batch_output, len(sub_patches), axis=0)
            batch_info = np.split(np.array(patches_info), len(patches_info), axis=0)
            batch_output_combined = list(
                zip(batch_info, batch_output))
            pred_list.extend(batch_output_combined)
            ###
            nr_proc += len(sub_patches)
            sys.stdout.write("\rProcessing Batch (%d/%d) of Tile (%d/%d)" %
                             (nr_proc, batch_count, tile_idx+1, nr_tiles))
            sys.stdout.flush()

        return pred_list

    def __select_valid_patches(self, patch_info_list, has_output_info=True):
        """Select valid patches coordinates from a set of input coordinates

        Args:
            patch_info_list: input patch coordinates
            has_output_info: whether output coordintes are available
        
        Return:
            sub_patch_info_list: list of valid patch coordinates
        
        """
        down_sample_ratio = self.wsi_mask.shape[0] / self.wsi_proc_shape[0]
        selected_indices = []
        for idx in range(patch_info_list.shape[0]):
            patch_info = patch_info_list[idx]
            patch_info = np.squeeze(patch_info)
            # get the box at corresponding mag of the mask
            if has_output_info:
                output_bbox = patch_info[1] * down_sample_ratio
            else:
                output_bbox = patch_info * down_sample_ratio
            output_bbox = np.rint(output_bbox).astype(np.int64)
            # coord of the output of the patch (i.e centre regions)
            output_roi = self.wsi_mask[
                output_bbox[0][0] : output_bbox[1][0],
                output_bbox[0][1] : output_bbox[1][1],
            ]
            if np.sum(output_roi) > 0:
                selected_indices.append(idx)
        sub_patch_info_list = patch_info_list[selected_indices]
        return sub_patch_info_list

    def __gen_prediction(self, tile_info_list, patch_info_list):
        """Generate prediction for a set of input tiles

        Args:
            tile_info_list: coordinates of input tiles
            patch_info_list: coordinates of input patches

        """ 
        wsi_pred_map_mmap_path = "%s/prob_map.npy" % self.cache_dir

        masking = lambda x, a, b: (a <= x) & (x <= b)
        nr_tiles = tile_info_list.shape[0]
        for idx in range(nr_tiles):
            tile_info = tile_info_list[idx]
            # select patch basing on top left coordinate of input
            start_coord = tile_info[0, 0]
            end_coord = tile_info[0, 1] - self.patch_input_shape_tmp
            selection = masking(
                patch_info_list[:, 0, 0, 0], start_coord[0], end_coord[0]
            ) & masking(patch_info_list[:, 0, 0, 1], start_coord[1], end_coord[1])
            tile_patch_info_list = np.array(patch_info_list[selection])

            # further select only the patches within the provided mask
            tile_patch_info_list = self.__select_valid_patches(tile_patch_info_list)

            # there no valid patches, so flush 0 and skip
            if tile_patch_info_list.shape[0] == 0:
                assemble_and_flush(wsi_pred_map_mmap_path, tile_info,
                               self.base_mag_ds, None)
                continue

            # change the coordinates from wrt slide to wrt tile
            tile_patch_info_list -= tile_info[:, 0]
            tile_data = self.wsi_handler.read_region(
                tile_info[0][0][::-1],
                self.wsi_proc_lvl,
                (tile_info[0][1] - tile_info[0][0])[::-1],
            )
            tile_data = np.array(tile_data)[..., :3]
            np.save("%s/cache_tile.npy" % self.cache_dir, tile_data)

            patch_output_list = self.__run_inference(
                tile_patch_info_list[:, 0, 0], idx, nr_tiles
            )

            assemble_and_flush(wsi_pred_map_mmap_path, tile_info,
                               self.base_mag_ds, patch_output_list)
        return

    def __dispatch_post_processing(self, tile_info_list, callback):
        """Initialise post processing

        Args:
            tile_info_list: coordinate information of WSI tiles
            callback: type of post processing callback used
        
        """
        proc_pool = None
        if self.postproc_workers > 0:
            proc_pool = Pool(processes=self.postproc_workers)

        wsi_pred_map_mmap_path = "%s/prob_map.npy" % self.cache_dir
        for idx in list(range(tile_info_list.shape[0])):
            tile_tl = tile_info_list[idx][0]
            tile_br = tile_info_list[idx][1]

            tile_info = (idx, tile_tl, tile_br)
            func_kwargs = {
                "nr_types": self.nr_types,
                "return_dict": True,
                "return_probs": self.return_probs,
            }

            if proc_pool is not None:
                proc_pool.apply_async(
                    post_proc_para_wrapper,
                    callback=callback,
                    args=(wsi_pred_map_mmap_path, tile_info, func_kwargs,),
                )
            else:
                results = post_proc_para_wrapper(
                    wsi_pred_map_mmap_path, tile_info, func_kwargs
                )
                callback(results)
        if proc_pool is not None:
            proc_pool.close()
            proc_pool.join()
        return

    def load_wsi(self, filename):
        """Load a WSI and get information"""
        wsi_ext = filename.split(".")[-1]
        self.wsi_handler = get_wsi_handler(filename, wsi_ext)

        self.wsi_ds_lvl = self.wsi_handler.metadata["level_downsamples"][
            self.wsi_proc_lvl
        ]
        self.ds_factor_mask = (
            self.wsi_handler.metadata["magnification"][self.wsi_proc_lvl] / 1.25
        )

        base_mag = self.wsi_handler.metadata["base_mag"]
        self.base_mag_ds = int(round(40.0 / base_mag)) # if scanned at 20x, this will be 2

        self.wsi_proc_shape = self.wsi_handler.metadata["level_dims"][self.wsi_proc_lvl]
        self.wsi_proc_shape = np.array(self.wsi_proc_shape[::-1])  # to Y, X

    def process_wsi(self, filename):
        """Process an individual WSI. This function will:

        1) Load the WSI
        2) Read and save low resolution thumbnail
        3) Generate and save a tissue mask
        4) Initialise result memory maps
        5) Run inference in tissue region
        6) Run post processing
        7) Save the results as a JSON file

        Args:
            filename: WSI path to process

        """
        self.full_filename = self.input_dir + "/" + filename
        print(self.full_filename)
        self.load_wsi(self.full_filename)

        # read wsi at low res for tissue seg (fixed at 1.25x obj mag)
        self.wsi_lowres = self.wsi_handler.data["src"]["img"]
        cv2.imwrite(
            "%s/thumbnail.png" % self.output_dir_wsi,
            cv2.cvtColor(self.wsi_lowres, cv2.COLOR_BGR2RGB),
        )

        # get tissue mask
        self.wsi_mask = get_tissue_mask(self.wsi_lowres)
        cv2.imwrite("%s/mask.png" % self.output_dir_wsi, self.wsi_mask * 255)

        # Initialise memory maps to prevent large arrays being stored in RAM
        out_ch = self.nr_types + 3 # nr types + instance channels
        self.wsi_prob_map_mmap = np.lib.format.open_memmap(
            "%s/prob_map.npy" % self.cache_dir,
            mode="w+",
            shape=tuple(self.wsi_proc_shape*self.base_mag_ds) + (out_ch,),
            dtype=np.float32,
        )
        self.wsi_inst_map = np.lib.format.open_memmap(
            "%s/pred_inst.npy" % self.cache_dir,
            mode="w+",
            shape=tuple(self.wsi_proc_shape*self.base_mag_ds),
            dtype=np.int32,
        )

        # ------------------------------RUN INFERENCE--------------------------------
        start = time.perf_counter() # init inference timer
        # the WSI is processed tile by tile. Get the coordinates of tiles and patches

        patch_input_shape_tmp = np.array(self.patch_input_shape)
        self.patch_input_shape_tmp = (
            patch_input_shape_tmp/self.base_mag_ds).astype('int')

        patch_stride = np.array(self.patch_output_shape)
        patch_stride = (patch_stride/self.base_mag_ds).astype('int')

        # make tile smaller if not at 40x. We extract more patches and resize if 20x
        inf_tile_input_shape = np.array(self.inf_tile_shape) / self.base_mag_ds
        inf_tile_info_list, patch_info_list = get_tile_patch_info(
            self.wsi_proc_shape,
            inf_tile_input_shape,
            self.patch_input_shape_tmp,
            patch_stride,
        )
        # get the predictions of each patch in a tile and save to memory map
        self.__gen_prediction(inf_tile_info_list, patch_info_list)
        print(". All Tiles Processed!")

        end = time.perf_counter()
        print("Inference Time:", round(end-start), 'secs')

        # --------------------------RUN POST PROCESSING-----------------------------
        # * define post processing callbacks - can only receive one argument
        def postproc_standard_callback(args):
            """Standard post processing callback. 

            Args:
                args: input to the callback

            """
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()
                return

            top_left = pos_args[1][::-1]

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())
            for inst_id, inst_info in inst_info_dict.items():
                # now correct the coordinate wrt wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            self.wsi_inst_map[
                tile_tl[0]: tile_br[0], tile_tl[1]: tile_br[1]
            ] = pred_inst

            pbar.update()
            return

        def postproc_fixborder_callback(args):
            """Callback for fixing incorrectly processed nuclei at the boundary. 
            For this, we define an `ambiguous region`, where predicted nuclei that are
            a certain number of pixels from the border are considered `border nuclei`.
            This callback replaces border nuclei with valid predictions.

            Args:
                args: input to the callback

            """
            results, pos_args = args
            run_idx, tile_tl, tile_br = pos_args
            pred_inst, inst_info_dict = results

            if len(inst_info_dict) == 0:
                pbar.update()
                return

            top_left = pos_args[1][::-1]

            wsi_max_id = 0
            if len(self.wsi_inst_info) > 0:
                wsi_max_id = max(self.wsi_inst_info.keys())

            # * exclude ambiguous output from old prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_inst = self.wsi_inst_map[
                tile_tl[0]: tile_br[0], tile_tl[1]: tile_br[1]
            ]
            roi_inst = np.copy(roi_inst)
            roi_edge = np.concatenate(
                [roi_inst[[0, -1], :].flatten(), roi_inst[:, [0, -1]].flatten()]
            )
            roi_boundary_inst_list = np.unique(
                roi_edge)[1:]  # exclude background
            roi_inner_inst_list = np.unique(roi_inst)[1:]
            roi_inner_inst_list = np.setdiff1d(
                roi_inner_inst_list, roi_boundary_inst_list, assume_unique=True
            )
            roi_inst = remove_inst(roi_inst, roi_inner_inst_list)
            self.wsi_inst_map[
                tile_tl[0]: tile_br[0], tile_tl[1]: tile_br[1]
            ] = roi_inst
            for inst_id in roi_inner_inst_list:
                self.wsi_inst_info.pop(inst_id, None)

            # * exclude unambiguous output from new prediction map
            # check 1 pix of 4 edges to find nuclei split at boundary
            roi_edge = pred_inst[roi_inst > 0]  # remove all overlap
            boundary_inst_list = np.unique(
                roi_edge)  # no background to exclude
            inner_inst_list = np.unique(pred_inst)[1:]
            inner_inst_list = np.setdiff1d(
                inner_inst_list, boundary_inst_list, assume_unique=True
            )
            pred_inst = remove_inst(pred_inst, boundary_inst_list)

            # * proceed to overwrite
            for inst_id in inner_inst_list:
                inst_info = inst_info_dict[inst_id]
                # now correct the coordinate wrt to wsi
                inst_info["bbox"] += top_left
                inst_info["contour"] += top_left
                inst_info["centroid"] += top_left
                self.wsi_inst_info[inst_id + wsi_max_id] = inst_info
            pred_inst[pred_inst > 0] += wsi_max_id
            pred_inst = roi_inst + pred_inst
            self.wsi_inst_map[
                tile_tl[0]: tile_br[0], tile_tl[1]: tile_br[1]
            ] = pred_inst

            pbar.update()
            return

        start = time.perf_counter() # init post processing timer

        proc_tile_shape = (np.array(self.proc_tile_shape)).astype(np.int64)
        proc_tile_coords = get_tile_info(
            self.wsi_proc_shape, proc_tile_shape, self.ambiguous_size
        )
        proc_grid_info, proc_boundary_info, proc_cross_info = proc_tile_coords
        proc_grid_info = self.__select_valid_patches(proc_grid_info, False)
        proc_boundary_info = self.__select_valid_patches(proc_boundary_info, False)
        proc_cross_info = self.__select_valid_patches(proc_cross_info, False)

        pbar_creator = lambda x, y: tqdm.tqdm(
            desc=y, leave=True, total=int(len(x)), ncols=80, ascii=True, position=0
        )
        pbar = pbar_creator(proc_grid_info, "Post Proc Phase 1")
        self.__dispatch_post_processing(proc_grid_info, postproc_standard_callback)
        pbar.close()
        pbar = pbar_creator(proc_boundary_info, "Post Proc Phase 2")
        self.__dispatch_post_processing(proc_boundary_info, postproc_fixborder_callback)
        pbar.close()
        pbar = pbar_creator(proc_cross_info, "Post Proc Phase 3")
        self.__dispatch_post_processing(proc_cross_info, postproc_fixborder_callback)
        pbar.close()

        # ! save as JSON because it isn't feasible to save the WSI at highest resolution
        json_dict = {}
        for inst_id, inst_info in self.wsi_inst_info.items():
            new_inst_info = {}
            for info_name, info_value in inst_info.items():
                # convert to JSON
                if isinstance(info_value, np.ndarray):
                    info_value = info_value.tolist()
                new_inst_info[info_name] = info_value
            json_dict[int(inst_id)] = new_inst_info
        with open(
            "%s/nuclei_dict.json" % self.output_dir_wsi, "w"
        ) as handle:
            json.dump(json_dict, handle)

        end = time.perf_counter()
        print("Post Proc Time:", round(end-start), 'secs')

    def load_model(self):
        """Loads the model and checkpoints"""
        print("Loading Model...")
        model_path = self.model_path
        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model=model_constructor(
                self.nr_types,
                self.patch_input_shape,
                self.patch_output_shape,
                self.input_norm,
            ),
            session_init=get_model_loader(model_path),
            input_names=self.input_tensor_names,
            output_names=self.output_tensor_names,
        )
        self.predictor = OfflinePredictor(pred_config)

    def load_filenames(self):
        """Get the list of all WSI files to process"""
        self.file_list = glob.glob("%s/*" % self.input_dir)
        self.file_list.sort()  # ensure same order

    def process_all_files(self):
        """Process each WSI and save results as JSON file"""

        if not os.path.exists(self.cache_dir):
            rm_n_mkdir(self.cache_dir)

        if not os.path.exists(self.output_dir):
            rm_n_mkdir(self.output_dir)

        for filename in self.file_list:
            filename = os.path.basename(filename)
            self.basename = os.path.splitext(filename)[0]
            self.output_dir_wsi = self.output_dir + "/" + self.basename
            # if not os.path.exists(self.output_dir_wsi):
            start = time.perf_counter()  # init timer

            rm_n_mkdir(self.output_dir_wsi)
            start_time_total = time.time()
            self.process_wsi(filename)

            end = time.perf_counter()
            print("Overall Time:", round(end-start), 'secs')


# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    args = docopt(__doc__, version="HoVer-Net Inference")
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args["--gpu"]

    # Raise exceptions for invalid / missing arguments
    if args["--model"] == None:
        raise Exception("A model path must be supplied as an argument with --model.")
    if args["--mode"] != "tile" and args["--mode"] != "wsi":
        raise Exception('Mode not recognised. Use either "tile" or "wsi"')
    if args["--input_dir"] == None:
        raise Exception(
            "An input directory must be supplied as an argument with --input_dir."
        )
    if args["--input_dir"] == args["--output_dir"]:
        raise Exception(
            "Input and output directories should not be the same- otherwise input directory will be overwritten."
        )

    if args["--mode"] == "tile":
        infer = InferTile()
        infer._parse_args(args)
        infer.load_model()
        infer.process_all_files()
    elif args["--mode"] == "wsi":  
        infer = InferWSI()
        infer._parse_args(args)
        infer.load_model()
        infer.load_filenames()
        infer.process_all_files()
