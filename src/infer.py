import argparse
import glob
import math
import os
from collections import deque

import cv2
import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt 

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from config import Config
from misc.utils import rm_n_mkdir
from misc.viz_utils import visualize_instances
import postproc.process_utils as proc_utils

import time
from time import sleep

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
            basename = filename.split('.')[0]
            print(self.inf_data_dir, basename, end=' ', flush=True)

            ##
            img = cv2.imread(self.inf_data_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ##
            pred_map = self.__gen_prediction(img, predictor)

            pred_inst, pred_type = proc_utils.process_instance(pred_map, type_classification=True, nr_types=self.nr_types)

            overlaid_output = visualize_instances(pred_inst, pred_type, img)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite('%s/%s.png' % (save_dir, basename), overlaid_output)
            np.save('%s/%s_inst.npy' % (save_dir, basename), pred_inst)
            np.save('%s/%s_type.npy' % (save_dir, basename), pred_type)
            print('FINISH')


####
class InferCoords(Config):

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
            pass
        else:
            self.wsiObj = ops.OpenSlide(self.full_filename)
            self.level_downsamples = self.wsiObj.level_downsamples
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = []
            # flipping cols into rows (Openslide to python format)
            for i in range(self.level_count):
                self.level_dimensions.append([self.wsiObj.level_dimensions[i][1], self.wsiObj.level_dimensions[i][0]])
    ####
    
    def tile_coords(self):
        '''
        Get the tile coordinates and dimensions for processing at level 0
        '''

        self.im_w = self.level_dimensions[self.proc_level][1]
        self.im_h = self.level_dimensions[self.proc_level][0]

        if self.nr_tiles_h > 1:
            step_h = math.floor(self.im_h / self.nr_tiles_h)
        if self.nr_tiles_w > 1:
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
        """
        # TODO Makeit parallel processing?!

        Extracts patches from the WSI before running inference. 
        If tissue mask is provided, only extract foreground patches  
        Tile is the tile number index       
        """
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

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
                win = self.read_region((int(col*self.ds_factor), int(row*self.ds_factor)),
                                    self.proc_level, (win_size[0], win_size[1]))                            
                if self.tissue is not None:
                    win_tiss = self.tissue[int(round(row/self.ds_factor_tiss)):int(round(row/self.ds_factor_tiss))+int(round(win_size[0]/self.ds_factor_tiss)),
                                      int(round(col/self.ds_factor_tiss)):int(round(col/self.ds_factor_tiss))+int(round(win_size[1]/self.ds_factor_tiss))]
                    tiss_vals = np.unique(win_tiss)
                    tiss_vals = tiss_vals.tolist()
                    if 1 in tiss_vals:
                        self.sub_patches.append(win)
                        self.patch_coords.append([row,col])
                    else:
                        self.skipped_idx.append(idx)
                else:
                    self.sub_patches.append(win)
                idx += 1
    ####

    def run_inference(self, tile):
        '''
        Run inference for extracted patches and apply post processing.
        Results are then assembled to the size of the original image.
        '''
        pred_map = deque()
        centroid_list = []
        type_list = []

        if len(self.sub_patches) > 0:
            while len(self.sub_patches) > self.inf_batch_size:
                mini_batch = self.sub_patches[:self.inf_batch_size]
                mini_batch2 = self.patch_coords[:self.inf_batch_size]
                self.sub_patches = self.sub_patches[self.inf_batch_size:]
                self.patch_coords = self.patch_coords[self.inf_batch_size:]
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
                mini_centroid_list = []
                mini_type_list = []
                for j in range(len(mini_output)):
                    # Post processing
                    patch_coords = mini_batch2[j]
                    summary_cent, summary_type = proc_utils.process_coords(
                        mini_output[j], self.type_classification, self.nr_types, patch_coords
                    )
                    
                    mini_centroid_list.extend(summary_cent)
                    mini_type_list.extend(summary_type)
                if len(mini_centroid_list) > 0:
                    centroid_list.extend(mini_centroid_list)
                    type_list.extend(mini_type_list)

            # Deal with the case when the number of patches is not divisisible by batch size
            if len(self.sub_patches) != 0:
                mini_output = self.predictor(self.sub_patches)[0]
                mini_output = np.split(mini_output, len(self.sub_patches), axis=0)
                mini_output_proc = []
                mini_centroid_list = []
                mini_type_list = []
                for j in range(len(mini_output)):
                    # Post processing
                    patch_coords = self.patch_coords[j]
                    summary_cent, summary_type = proc_utils.process_coords(
                        mini_output[j], self.type_classification, self.nr_types, patch_coords
                    )
                    
                    mini_centroid_list.extend(summary_cent)
                    mini_type_list.extend(summary_type)
                if len(mini_centroid_list) > 0:
                    centroid_list.extend(mini_centroid_list)
                    type_list.extend(mini_type_list)

        
        return centroid_list, type_list

    ####
    def process_wsi(self, filename):
        '''
        Process an individual WSI. This function will:
        1) Load the OpenSlide WSI object
        2) Generate the tissue mask
        3) Get tile coordinate info
        4) Extract patches from foreground regions
        4) Run inference and return centroids and class predictions
        '''
        filename = os.path.basename(filename)
        self.basename = filename.split('.')[0]
        print('Processing', self.basename, end='. ', flush=True)

        # Load the OpenSlide WSI object
        self.full_filename = self.inf_wsi_dir + filename
        self.load_wsi()

        self.ds_factor = self.level_downsamples[self.proc_level]
        self.ds_factor_tiss = self.level_downsamples[self.tiss_level] / self.level_downsamples[self.proc_level]

        if self.tissue_inf:
            # Generate tissue mask 
            ds_img = self.read_region(
                (0,0), 
                self.tiss_level, 
                (self.level_dimensions[self.tiss_level][1], self.level_dimensions[self.tiss_level][0])
                )
            self.tissue = proc_utils.get_tissue_mask(ds_img)

        # Coordinate info for tile processing 
        self.tile_coords()

        # Run inference tile by tile - if self.tissue_inf == True, only process tissue regions
        pred_map_list = []
        self.centroid_summary = []
        self.type_summary = []
        # Initialise progress bar
        bar = progressbar.ProgressBar(maxval=self.nr_tiles_h*self.nr_tiles_w, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for tile in range(len(self.tile_info)):
            bar.update(tile+1)
            sleep(0.1)

            self.extract_patches(tile)

            centroid_list, type_list = self.run_inference(tile)
            self.centroid_summary.extend(centroid_list)
            self.type_summary.extend(type_list)
            
        bar.finish()      
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
        rm_n_mkdir(self.save_dir) 

        start_time_total = time.time()
        for filename in self.file_list:
            self.process_wsi(filename)
        
            np.savez('%s/%s.npz' % (self.inf_output_dir, self.basename), centroid=self.centroid_summary, type=self.type_summary)
            end_time_total = time.time()
            print('FINISHED. Time: ', time_it(start_time_total, end_time_total), 'secs')

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
            pass
        else:
            self.wsiObj = ops.OpenSlide(self.full_filename)
            self.level_downsamples = self.wsiObj.level_downsamples
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = []
            # flipping cols into rows (Openslide to python format)
            for i in range(self.level_count):
                self.level_dimensions.append([self.wsiObj.level_dimensions[i][1], self.wsiObj.level_dimensions[i][0]])
    ####
    
    def tile_coords(self):
        '''
        Get the tile coordinates and dimensions for processing at level 0
        '''

        self.im_w = self.level_dimensions[self.proc_level][1]
        self.im_h = self.level_dimensions[self.proc_level][0]

        if self.nr_tiles_h > 1:
            step_h = math.floor(self.im_h / self.nr_tiles_h)
        if self.nr_tiles_w > 1:
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
        """
        # TODO Makeit parallel processing?!

        Extracts patches from the WSI before running inference. 
        If tissue mask is provided, only extract foreground patches  
        Tile is the tile number index       
        """
        step_size = self.infer_mask_shape
        msk_size = self.infer_mask_shape
        win_size = self.infer_input_shape

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

        # Generating sub-patches from WSI                     
        idx = 0
        for row in range(start_h, last_h, step_size[0]):
            for col in range(start_w, last_w, step_size[1]):  
                win = self.read_region((int(col*self.ds_factor), int(row*self.ds_factor)),
                                    self.proc_level, (win_size[0], win_size[1]))                            
                if self.tissue is not None:
                    win_tiss = self.tissue[int(round(row/self.ds_factor_tiss)):int(round(row/self.ds_factor_tiss))+int(round(win_size[0]/self.ds_factor_tiss)),
                                      int(round(col/self.ds_factor_tiss)):int(round(col/self.ds_factor_tiss))+int(round(win_size[1]/self.ds_factor_tiss))]
                    tiss_vals = np.unique(win_tiss)
                    tiss_vals = tiss_vals.tolist()
                    if 1 in tiss_vals:
                        self.sub_patches.append(win)
                    else:
                        self.skipped_idx.append(idx)
                else:
                    self.sub_patches.append(win)
                idx += 1
    ####

    def run_inference(self, tile, save_zeros):
        '''
        Run inference for extracted patches and apply post processing.
        Results are then assembled to the size of the original image.
        '''
        pred_map = deque()

        if len(self.sub_patches) > 0:
            while len(self.sub_patches) > self.inf_batch_size:
                mini_batch = self.sub_patches[:self.inf_batch_size]
                self.sub_patches = self.sub_patches[self.inf_batch_size:]
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, self.inf_batch_size, axis=0)
                mini_output_proc = []
                for j in range(len(mini_output)):
                    # Post processing
                    pred_inst, pred_type = proc_utils.process_instance(
                        mini_output[j], self.type_classification, self.nr_types
                    )
                    mini_output_proc.append(pred_type)
                pred_map.extend(mini_output_proc)

            # Deal with the case when the number of patches is not divisible by batch size
            if len(self.sub_patches) != 0:
                mini_output = self.predictor(self.sub_patches)[0]
                mini_output = np.split(mini_output, len(self.sub_patches), axis=0)
                mini_output_proc = []
                for j in range(len(mini_output)):
                    # Post processing
                    pred_inst, pred_type = proc_utils.process_instance(
                        mini_output[j], self.type_classification, self.nr_types
                    )
                    mini_output_proc.append(pred_type)
                pred_map.extend(mini_output_proc)

            output_patch_shape = np.squeeze(pred_map[0]).shape
            if self.tissue is not None:
                if len(output_patch_shape) != 2:
                    zeros = np.zeros(
                        (output_patch_shape[0], output_patch_shape[1], output_patch_shape[2]))
                else:
                    zeros = np.zeros(
                        (output_patch_shape[0], output_patch_shape[1]))
                for idx in self.skipped_idx:
                    pred_map.insert(idx, zeros)
                    self.skipped_idx = [x+1 for x in self.skipped_idx]
            ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

            #### Assemble back into full image
            pred_map = np.squeeze(np.array(pred_map))
            pred_map = np.reshape(
                pred_map, (self.nr_step_h, self.nr_step_w) + pred_map.shape[1:])
            pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                np.transpose(pred_map, [0, 2, 1, 3])
            pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1],
                                            pred_map.shape[2] * pred_map.shape[3], ch))

            # Crop back to original size
            pred_map = np.squeeze(pred_map[:self.im_h, :self.im_w])
            pred_map = pred_map.astype('uint8')
    
        else:
            if save_zeros == False:
                pred_map = None
            else:
                pred_map = np.zeros([self.im_h, self.im_w])
        
        return pred_map

    ####
    def process_wsi(self, filename):
        '''
        Process an individual WSI. This function will:
        1) Load the OpenSlide WSI object
        2) Generate the tissue mask
        3) Get tile coordinate info
        4) Extract patches from foreground regions
        4) Run inference and return centroids and class predictions
        '''
       
        print('Processing', self.basename, end='. ', flush=True)

        # Load the OpenSlide WSI object
        self.full_filename = self.inf_wsi_dir + filename
        print(self.full_filename)
        self.load_wsi()

        self.ds_factor = self.level_downsamples[self.proc_level]
        self.ds_factor_tiss = self.level_downsamples[self.tiss_level] / self.level_downsamples[self.proc_level]

        if self.tissue_inf:
            # Generate tissue mask 
            ds_img = self.read_region(
                (0,0), 
                self.tiss_level, 
                (self.level_dimensions[self.tiss_level][1], self.level_dimensions[self.tiss_level][0])
                )
            self.tissue = proc_utils.get_tissue_mask(ds_img)

        # Coordinate info for tile processing 
        self.tile_coords()

        # Run inference tile by tile - if self.tissue_inf == True, only process tissue regions
        pred_map_list = []
        self.centroid_summary = []
        self.type_summary = []
        # Initialise progress bar
        bar = progressbar.ProgressBar(maxval=self.nr_tiles_h*self.nr_tiles_w, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for tile in range(len(self.tile_info)):
            bar.update(tile+1)
            sleep(0.1)

            self.extract_patches(tile)   
            
            pred_map = self.run_inference(tile, save_zeros=False)
            if pred_map is not None:
                np.save('%s/%s/%s_%s.npy' % (self.inf_output_dir, self.basename, self.basename, str(tile)), pred_map)
                pred_map = None
                img_tile = self.read_region((self.tile_info[tile][0], self.tile_info[tile][1]),
                                    self.proc_level, (self.tile_info[tile][2], self.tile_info[tile][3])) 
                plt.imsave('%s/%s/%s_%s.png' % (self.inf_output_dir, self.basename, self.basename, str(tile)), img_tile)
                img_tile = None
            
        bar.finish()      
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
        #rm_n_mkdir(self.save_dir) 

        for filename in self.file_list:
            filename = os.path.basename(filename)
            self.basename = filename.split('.')[0]
            rm_n_mkdir(self.save_dir + '/' + self.basename)
            start_time_total = time.time()
            self.process_wsi(filename)
            end_time_total = time.time()
            print('FINISHED. Time: ', time_it(start_time_total, end_time_total), 'secs')
        

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', help='Use either "roi_seg" or "wsi_coords".')
    args = parser.parse_args()
        
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    # Import libraries for WSI processing
    if args.mode.split('_')[0] == 'wsi':
        import openslide as ops 
        import progressbar
        # import matlab
        # from matlab import engine

    if args.mode == 'roi_seg':
        infer = InferROI()
        infer.run() 
    elif args.mode == 'wsi_coords':
        infer = InferCoords()
        infer.load_model() 
        infer.load_filenames()
        infer.process_all_wsi() 
    elif args.mode == 'wsi_seg': # Currently this saves each individual tile - need to optimise
        infer = InferWSI()
        infer.load_model() 
        infer.load_filenames()
        infer.process_all_wsi()  
    else:
        print('Mode not recognised. Use either "roi_seg" or "wsi_coords"')
