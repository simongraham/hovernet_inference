import argparse
import glob
import math
import os
import sys
from collections import deque

import cv2
import numpy as np

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from config import Config
from misc.utils import rm_n_mkdir
from misc.viz_utils import visualize_instances
import postproc.process_utils as proc_utils

import time

# disable logging info
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from tensorflow import logging
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorpack import logger
logger._getlogger().disabled = True 

####

def time_it(start_time, end_time):
    diff = end_time - start_time
    return str(round(diff))
####

class InferROI(Config):

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x : input image to be segmented. It will be split into patches
                to run the prediction upon before being assembled back            
        """    
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)
        
        im_h = x.shape[0] 
        im_w = x.shape[1]

        last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
        last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

        diff_h = win_size[0] - step_size[0]
        padt = diff_h // 2
        padb = last_h + win_size[0] - im_h

        diff_w = win_size[1] - step_size[1]
        padl = diff_w // 2
        padr = last_w + win_size[1] - im_w

        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'reflect')

        #### TODO: optimize this
        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0], 
                        col:col+win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.inf_batch_size:
            mini_batch  = sub_patches[:self.inf_batch_size]
            sub_patches = sub_patches[self.inf_batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        #### Assemble back into full image
        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        #### Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map

    ####
    def run(self):

        model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(),
            session_init = get_model_loader(model_path),
            input_names  = self.eval_inf_input_tensor_names,
            output_names = self.eval_inf_output_tensor_names)
        predictor = OfflinePredictor(pred_config)

        save_dir = self.inf_output_dir
        file_list = glob.glob('%s/*%s' % (self.inf_data_dir, self.inf_imgs_ext))
        file_list.sort() # ensure same order

        rm_n_mkdir(save_dir)       
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = os.path.splitext(filename)[0]
            print(self.inf_data_dir, basename, end=' ', flush=True)

            ###
            img = cv2.imread(self.inf_data_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ###
            pred_map = self.__gen_prediction(img, predictor)

            pred_inst, pred_type = proc_utils.process_instance(pred_map, type_classification=True, nr_types=self.nr_types)

            overlaid_output = visualize_instances(pred_inst, pred_type, img)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite('%s/%s.png' % (save_dir, basename), overlaid_output)
            np.save('%s/%s_inst.npy' % (save_dir, basename), pred_inst)
            if pred_type is not None:
                np.save('%s/%s_type.npy' % (save_dir, basename), pred_type)
            print('FINISH')
####

class InferWSI(Config):

    def read_region(self, location, level, patch_size):
        '''
        Loads a patch from an OpenSlide object
        '''
        if self.inf_wsi_ext == '.jp2':
            y1 = int(location[1] / pow(2, level)) + 1
            x1 = int(location[0] / pow(2, level)) + 1
            y2 = int(y1 + patch_size[1] -1)
            x2 = int(x1 + patch_size[0] -1)
            # this will read patch using matlab engine
            patch = self.wsiObj.read_region(self.full_filename, level, matlab.int32([x1,x2,y1,y2]))
            patch = np.array(patch._data).reshape(patch.size, order='F')
        else:
            patch = self.wsiObj.read_region(location, level, patch_size)
            r, g, b, _ = cv2.split(np.array(patch))
            patch = cv2.merge([r, g, b])
        return patch
    
    def load_wsi(self):
        '''
        Load WSI using OpenSlide. Note, if using JP2, appropriate
        matlab scripts need to be placed in the working directory
        '''
        if self.inf_wsi_ext == '.jp2':
            try:
                self.wsiObj = engine.start_matlab()
            except:
                print ("Matlab Engine not started...")
            self.wsiObj.cd(os.getcwd(), nargout=0)
            level_dim, level_downsamples, level_count  = self.wsiObj.JP2Image(self.full_filename, nargout=3)
            level_dim = np.float32(level_dim)
            self.level_dimensions = level_dim.tolist()
            self.level_count = np.int32(level_count)
            level_downsamples = np.float32(level_downsamples)
            self.level_downsamples = []
            for i in range(self.level_count):
                self.level_downsamples.append(level_downsamples[i][0])
            self.scan_resolution = [0.275, 0.275]  # scan resolution of the Omnyx scanner at UHCW
        else:
            self.wsiObj = ops.OpenSlide(self.full_filename)
            self.level_downsamples = self.wsiObj.level_downsamples
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = []
            # flipping cols into rows (Openslide to python format)
            for i in range(self.level_count):
                self.level_dimensions.append([self.wsiObj.level_dimensions[i][1], self.wsiObj.level_dimensions[i][0]])
            self.scan_resolution = [float(self.wsiObj.properties.get('openslide.mpp-x')),
                                    float(self.wsiObj.properties.get('openslide.mpp-y'))]
    ####
    
    def tile_coords(self):
        '''
        Get the tile coordinates and dimensions for processing at level 0
        '''

        self.im_w = self.level_dimensions[self.proc_level][1]
        self.im_h = self.level_dimensions[self.proc_level][0]

        if self.nr_tiles_h > 0:
            step_h = math.floor(self.im_h / self.nr_tiles_h)
        if self.nr_tiles_w > 0:
            step_w = math.floor(self.im_w / self.nr_tiles_w)

        self.tile_info = []

        for row in range(self.nr_tiles_h):
            for col in range(self.nr_tiles_w):
                start_h = row*step_h
                start_w = col*step_w
                if row == self.nr_tiles_h - 1:
                    extra_h = self.im_h - (self.nr_tiles_h * step_h)
                    dim_h = step_h + extra_h
                else:
                    dim_h = step_h 
                if col == self.nr_tiles_w - 1:
                    extra_w = self.im_w - (self.nr_tiles_w * step_w)
                    dim_w = step_w + extra_w
                else:
                    dim_w = step_w
                self.tile_info.append((int(start_w), int(start_h), int(dim_w), int(dim_h)))
    ####

    def extract_patches(self, tile):
        '''
        # TODO Make it parallel processing?!

        Extracts patches from the WSI before running inference.
        If tissue mask is provided, only extract foreground patches
        Tile is the tile number index
        '''
        step_size = np.array(self.infer_mask_shape)
        msk_size = np.array(self.infer_mask_shape)
        win_size = np.array(self.infer_input_shape)
        if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
            step_size = np.int64(step_size / 2)
            msk_size = np.int64(msk_size / 2)
            win_size = np.int64(win_size / 2)

        def get_last_steps(length, msk_size, step_size):
            nr_step = math.ceil((length - msk_size) / step_size)
            last_step = (nr_step + 1) * step_size
            return int(last_step), int(nr_step + 1)

        last_h, self.nr_step_h = get_last_steps(self.tile_info[tile][3], msk_size[0], step_size[0])
        last_w, self.nr_step_w = get_last_steps(self.tile_info[tile][2], msk_size[1], step_size[1])

        start_h = self.tile_info[tile][1]
        start_w = self.tile_info[tile][0]
        last_h += start_h
        last_w += start_w

        self.sub_patches = []
        self.skipped_idx = []
        self.patch_coords = []

        # Generating sub-patches from WSI
        idx = 0
        for row in range(start_h, last_h, step_size[0]):
            for col in range(start_w, last_w, step_size[1]):
                if self.tissue is not None:
                    win_tiss = self.tissue[
                               int(round(row / self.ds_factor_tiss)):int(round(row / self.ds_factor_tiss)) + int(
                                   round(win_size[0] / self.ds_factor_tiss)),
                               int(round(col / self.ds_factor_tiss)):int(round(col / self.ds_factor_tiss)) + int(
                                   round(win_size[1] / self.ds_factor_tiss))]
                    if np.sum(win_tiss) > 0:
                        self.patch_coords.append([row, col])
                    else:
                        self.skipped_idx.append(idx)
                else:
                    self.patch_coords.append([row, col])
                idx += 1
    ####

    def load_batch(self, batch_coor):
        batch = []
        win_size = self.infer_input_shape
        if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
            win_size = np.int64(np.array(self.infer_input_shape)/2)

        for coor in batch_coor:
            win = self.read_region((int(coor[1] * self.ds_factor), int(coor[0] * self.ds_factor)),
                                   self.proc_level, (win_size[0], win_size[1]))
            if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
                win = cv2.resize(win, (win.shape[1]*2, win.shape[0]*2), cv2.INTER_LINEAR) # cv.INTER_LINEAR is good for zooming
            batch.append(win)
        return batch
    ####

    def run_inference(self, tile):
        '''
        Run inference for extracted patches and apply post processing.
        Results are then assembled to the size of the original image.
        '''

        pred_map = deque()
        mask_list = []
        type_list = []
        cent_list = []
        offset = (self.infer_input_shape[0] - self.infer_mask_shape[0]) / 2
        idx = 0
        batch_count = np.floor(len(self.patch_coords) / self.inf_batch_size)

        if len(self.patch_coords) > 0:
            while len(self.patch_coords) > self.inf_batch_size:
                # print('Batch(%d/%d) of Tile(%d/%d)' % (idx+1, batch_count, tile+1, self.nr_tiles_h*self.nr_tiles_w ))
                sys.stdout.write("\rBatch(%d/%d) of Tile(%d/%d)" % (
                idx + 1, batch_count, tile + 1, self.nr_tiles_h * self.nr_tiles_w))
                sys.stdout.flush()
                idx += 1
                mini_batch_coor = self.patch_coords[:self.inf_batch_size]
                mini_batch = self.load_batch(mini_batch_coor)
                self.patch_coords = self.patch_coords[self.inf_batch_size:]
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
                mini_mask_list = []
                mini_type_list = []
                mini_cent_list = []
                for j in range(len(mini_output)):
                    # Post processing
                    patch_coords = mini_batch_coor[j]
                    mask_list_tmp, type_list_tmp, cent_list_tmp = proc_utils.process_instance_wsi(
                        mini_output[j], self.type_classification, self.nr_types, patch_coords, offset=offset,
                        scan_resolution=self.scan_resolution[0]
                    )
                    mini_mask_list.extend(mask_list_tmp)
                    mini_type_list.extend(type_list_tmp)
                    mini_cent_list.extend(cent_list_tmp)
                if len(mini_cent_list) > 0:
                    mask_list.extend(mini_mask_list)
                    type_list.extend(mini_type_list)
                    cent_list.extend(mini_cent_list)

            # Deal with the case when the number of patches is not divisisible by batch size
            if len(self.patch_coords) != 0:
                mini_batch = self.load_batch(self.patch_coords)
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, len(self.patch_coords), axis=0)
                mini_mask_list = []
                mini_type_list = []
                mini_cent_list = []
                for j in range(len(mini_output)):
                    # Post processing
                    patch_coords = self.patch_coords[j]
                    mask_list_tmp, type_list_tmp, cent_list_tmp = proc_utils.process_instance_wsi(
                        mini_output[j], self.type_classification, self.nr_types, patch_coords, offset=offset,
                        scan_resolution=self.scan_resolution[0]
                    )
                    mini_mask_list.extend(mask_list_tmp)
                    mini_type_list.extend(type_list_tmp)
                    mini_cent_list.extend(cent_list_tmp)
                if len(mini_cent_list) > 0:
                    mask_list.extend(mini_mask_list)
                    type_list.extend(mini_type_list)
                    cent_list.extend(mini_cent_list)
        else:
            mask_list = None
            type_list = None
            cent_list = None

        return mask_list, type_list, cent_list
        ####

    def process_wsi(self, filename):
        '''
        Process an individual WSI. This function will:
        1) Load the OpenSlide WSI object
        2) Generate the tissue mask
        3) Get tile coordinate info
        4) Extract patches from foreground regions
        5) Run inference and return npz for each tile of 
           masks, type predictions and centroid locations
        '''
        # Load the OpenSlide WSI object
        self.full_filename = self.inf_wsi_dir + filename
        print(self.full_filename)
        self.load_wsi()

        self.ds_factor = self.level_downsamples[self.proc_level]
        self.ds_factor_tiss = self.level_downsamples[self.tiss_level] / self.level_downsamples[self.proc_level]

        is_valid_tissue_level = True
        tissue_level = self.tiss_level
        if tissue_level < len(self.level_downsamples):  # if given tissue level exist
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_level]
        elif len(self.level_downsamples) > 1:
            tissue_level = len(self.level_downsamples) - 1  # to avoid tissue segmentation at level 0
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_level]
        else:
            is_valid_tissue_level = False

        if self.tissue_inf & is_valid_tissue_level:
            # Generate tissue mask
            ds_img = self.read_region(
                (0, 0),
                tissue_level,
                (self.level_dimensions[tissue_level][1], self.level_dimensions[tissue_level][0])
            )

            # downsampling factor if image is largest dimension of the image is greater than 5000 at given tissue level
            # to reduce tissue segmentation time
            proc_scale = 1 / np.ceil(np.max(ds_img.shape) / 5000)

            self.tissue = proc_utils.get_tissue_mask(ds_img, proc_scale)

        # Coordinate info for tile processing
        self.tile_coords()

        # Run inference tile by tile - if self.tissue_inf == True, only process tissue regions
        for tile in range(len(self.tile_info)):

            self.extract_patches(tile)

            mask_list, type_list, cent_list = self.run_inference(tile)

            # Only save files where there exists nuclei
            if mask_list is not None:
                np.savez('%s/%s/%s_%s.npz' % (self.inf_output_dir, self.basename, self.basename, str(tile)),
                         mask=mask_list, type=type_list, centroid=cent_list)

                # bar.finish()
    ####

    def load_model(self):
        '''
        Loads the model and checkpoints according to the model stated in config.py
        '''
        model_path = self.inf_model_path

        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(),
            session_init = get_model_loader(model_path),
            input_names  = self.eval_inf_input_tensor_names,
            output_names = self.eval_inf_output_tensor_names)
        self.predictor = OfflinePredictor(pred_config)
    ####

    def load_filenames(self):
        '''
        Get the list of all WSI files to process
        '''
        self.file_list = glob.glob('%s/*%s' % (self.inf_wsi_dir, self.inf_wsi_ext))
        self.file_list.sort() # ensure same order
####
    
    def process_all_wsi(self):
        '''
        Process each WSI one at a time and save results as npz file
        '''
        self.save_dir = self.inf_output_dir

        for filename in self.file_list:
            filename = os.path.basename(filename)
            self.basename = os.path.splitext(filename)[0]
            if os.path.isdir(os.path.join(self.inf_output_dir, 'temp', self.basename)):
                continue
            os.makedirs(os.path.join(self.inf_output_dir, 'temp', self.basename), exist_ok=True)
            rm_n_mkdir(self.save_dir + '/' + self.basename)
            start_time_total = time.time()
            self.process_wsi(filename)
            end_time_total = time.time()
            print('FINISHED. Time: ', time_it(start_time_total, end_time_total), 'secs')
        
####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', help='Use either "roi" or "wsi".')
    args = parser.parse_args()
        
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        args.gpu = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    if (args.mode != 'wsi') & (args.mode != 'roi'):
        args.mode = 'roi'

    # Import libraries for WSI processing
    if args.mode == 'wsi':
        import openslide as ops 

        try:
            import matlab
            from matlab import engine
        except:
            pass

    if args.mode == 'roi':
        infer = InferROI()
        infer.run() 
    elif args.mode == 'wsi': # currently saves results per tile
        infer = InferWSI()
        infer.load_model() 
        infer.load_filenames()
        infer.process_all_wsi() 
    else:
        print('Mode not recognised. Use either "roi" or "wsi"')
