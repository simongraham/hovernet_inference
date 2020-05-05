"""run.

Usage:
  run.py [--gpu=<id>] [--mode=<mode>] [--model=<path>] [--batch_size=<n>] [--input_dir=<path>] [--output_dir=<path>] [--tiles_h=<n>] [--tiles_w=<n>] [--return_masks]
  run.py (-h | --help)
  run.py --version

Options:
  -h --help            Show this string.
  --version            Show version.
  --gpu=<id>           GPU list. [default: 0]
  --mode=<mode>        Inference mode. 'roi' or 'wsi'. [default: roi]
  --model=<path>       Path to saved checkpoint.
  --input_dir=<path>   Directory containing input images/WSIs.
  --output_dir=<path>  Directory where the output will be saved. [default: output/]
  --batch_size=<n>     Batch size. [default: 25]
  --tiles_h=<n>        Number of tile in vertical direction for WSI processing. [default: 3]
  --tiles_w=<n>        Number of tiles in horizontal direction for WSI processing. [default: 3]
  --return_masks       Whether to return cropped nuclei masks
"""


from docopt import docopt
import glob
import math
import os
import sys
import importlib
from collections import deque

import cv2
import numpy as np

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils.sessinit import get_model_loader

from tensorpack import logger
logger._getlogger().disabled = True # disable logging of network info

from hover.misc.utils import rm_n_mkdir
from hover.misc.viz_utils import visualize_instances
import hover.postproc.process_utils as proc_utils

import time

####
def time_it(start_time, end_time):
    """
    Helper function to compute run time
    """

    diff = end_time - start_time
    return str(round(diff))
####

class InferROI(object):
    def __init__(self,):
        self.nr_types = 6  # denotes number of classes (including BG) for nuclear type classification
        self.input_shape = [256, 256]
        self.mask_shape = [164, 164] 
        self.input_norm  = True # normalize RGB to 0-1 range

        # for inference during evalutation mode i.e run by infer.py
        self.input_tensor_names = ['images']
        self.output_tensor_names = ['predmap-coded']

    def load_params(self, args):
        """
        Load arguments
        """
        # Paths
        self.model_path  = args['--model']
        self.input_dir = args['--input_dir']
        self.output_dir = args['--output_dir']

        # Processing
        self.batch_size = int(args['--batch_size'])
    
    def get_model(self):
        model_constructor = importlib.import_module('hover.model.graph')
        model_constructor = model_constructor.Model_NP_HV  
        return model_constructor # NOTE return alias, not object

    def __gen_prediction(self, x, predictor):
        """
        Using 'predictor' to generate the prediction of image 'x'

        Args:
            x        : input image to be segmented. It will be split into patches
                       to run the prediction upon before being assembled back 
            predictor: A predictor built from a given config.           
        """    

        step_size = self.mask_shape
        msk_size = self.mask_shape
        win_size = self.input_shape

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

        sub_patches = []
        # generating subpatches from orginal
        for row in range(0, last_h, step_size[0]):
            for col in range (0, last_w, step_size[1]):
                win = x[row:row+win_size[0], 
                        col:col+win_size[1]]
                sub_patches.append(win)

        pred_map = deque()
        while len(sub_patches) > self.batch_size:
            mini_batch  = sub_patches[:self.batch_size]
            sub_patches = sub_patches[self.batch_size:]
            mini_output = predictor(mini_batch)[0]
            mini_output = np.split(mini_output, self.batch_size, axis=0)
            pred_map.extend(mini_output)
        if len(sub_patches) != 0:
            mini_output = predictor(sub_patches)[0]
            mini_output = np.split(mini_output, len(sub_patches), axis=0)
            pred_map.extend(mini_output)

        output_patch_shape = np.squeeze(pred_map[0]).shape
        ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

        # Assemble back into full image
        pred_map = np.squeeze(np.array(pred_map))
        pred_map = np.reshape(pred_map, (nr_step_h, nr_step_w) + pred_map.shape[1:])
        pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                        np.transpose(pred_map, [0, 2, 1, 3])
        pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                         pred_map.shape[2] * pred_map.shape[3], ch))
        pred_map = np.squeeze(pred_map[:im_h,:im_w]) # just crop back to original size

        return pred_map

    ####
    def load_model(self):
        print('Loading Model...')
        model_path = self.model_path
        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(self.nr_types, self.input_shape, self.mask_shape, self.input_norm),
            session_init = get_model_loader(model_path),
            input_names  = self.input_tensor_names,
            output_names = self.output_tensor_names)
        self.predictor = OfflinePredictor(pred_config)

    def process(self):
        """
        Process image files within a directory.
        For each image, the function will:
        1) Load the image
        2) Extract patches the entire image
        3) Run inference
        4) Return output numpy file and overlay
        """

        save_dir = self.output_dir
        file_list = glob.glob('%s/*' %self.input_dir)
        file_list.sort() # ensure same order

        rm_n_mkdir(save_dir)       
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = os.path.splitext(filename)[0]
            print(self.input_dir, basename, end=' ', flush=True)
            print(filename)

            ###
            img = cv2.imread(self.input_dir + '/' + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            ###
            pred_map = self.__gen_prediction(img, self.predictor)

            pred_inst, pred_type = proc_utils.process_instance(pred_map, nr_types=self.nr_types)
            
            overlaid_output = visualize_instances(pred_inst, pred_type, img)
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)

            # combine instance and type arrays for saving
            pred_inst = np.expand_dims(pred_inst, -1)
            pred_type = np.expand_dims(pred_type, -1)
            pred = np.dstack([pred_inst, pred_type])

            cv2.imwrite('%s/%s.png' % (save_dir, basename), overlaid_output)
            np.save('%s/%s.npy' % (save_dir, basename), pred)
####

class InferWSI(object):
    def __init__(self,):
        self.nr_types = 6  # denotes number of classes (including BG) for nuclear type classification
        self.input_shape = [256, 256]
        self.mask_shape = [164, 164] 
        self.input_norm  = True # normalize RGB to 0-1 range
        self.proc_lvl   = 0 # WSI level at which to process
        self.tiss_seg = True # only process tissue areas
        self.tiss_lvl = 3 # WSI level at which perform tissue segmentation 

        # for inference during evalutation mode i.e run by infer.py
        self.input_tensor_names = ['images']
        self.output_tensor_names = ['predmap-coded']

    def load_params(self, args):
        """
        Load arguments
        """

        # Paths
        self.model_path  = args['--model']
        # get absolute path for input directory - otherwise may give error in JP2Image.m
        self.input_dir = os.path.abspath(args['--input_dir'])
        self.output_dir = args['--output_dir']

        # Processing
        self.batch_size = int(args['--batch_size'])
        # Below specific to WSI processing
        self.nr_tiles_h = int(args['--tiles_h'])
        self.nr_tiles_w = int(args['--tiles_w'])
        self.return_masks = args['--return_masks']
    
    def get_model(self):
        model_constructor = importlib.import_module('hover.model.graph')
        model_constructor = model_constructor.Model_NP_HV  
        return model_constructor # NOTE return alias, not object

    def read_region(self, location, level, patch_size, wsi_ext):
        """
        Loads a patch from an OpenSlide object
        
        Args:
            location: top left coordinates of patch
            level: level of WSI pyramid at which to extract
            patch_size: patch size to extract
            wsi_ext: WSI file extension
        
        Returns:
            patch: extracted patch (np array)
        """

        if wsi_ext == 'jp2':
            x1 = int(location[0] / pow(2, level)) + 1
            y1 = int(location[1] / pow(2, level)) + 1
            x2 = int(x1 + patch_size[0] -1)
            y2 = int(y1 + patch_size[1] -1)
            # this will read patch using matlab engine
            patch = self.wsiObj.read_region(self.full_filename, level, matlab.int32([y1,y2,x1,x2]))
            patch = np.array(patch._data).reshape(patch.size, order='F')
        else:
            patch = self.wsiObj.read_region(location, level, patch_size)
            r, g, b, _ = cv2.split(np.array(patch))
            patch = cv2.merge([r, g, b])
        return patch
    
    def load_wsi(self, wsi_ext):
        """
        Load WSI using OpenSlide. Note, if using JP2, appropriate
        matlab scripts need to be placed in the working directory

        Args:
            wsi_ext: file extension of the whole-slide image
        """

        if wsi_ext == 'jp2':
            try:
                self.wsiObj = engine.start_matlab()
            except:
                print ("Matlab Engine not started...")
            self.wsiObj.cd(os.getcwd() + '/hover', nargout=0)
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
        """
        Get the tile coordinates and dimensions for processing at level 0
        """

        self.im_w = self.level_dimensions[self.proc_lvl][1]
        self.im_h = self.level_dimensions[self.proc_lvl][0]

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
        """
        Extracts patches from the WSI before running inference.
        If tissue mask is provided, only extract foreground patches.

        Args:
            tile: tile number index
        """
        
        step_size = np.array(self.mask_shape)
        msk_size = np.array(self.mask_shape)
        win_size = np.array(self.input_shape)
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
        self.patch_idx = []
        self.patch_coords = []

        # Generating sub-patches from WSI
        idx = 0
        for row in range(start_h, last_h, step_size[0]):
            for col in range(start_w, last_w, step_size[1]):
                if self.tiss_seg is True:
                    win_tiss = self.tissue[
                               int(round(row / self.ds_factor_tiss)):int(round(row / self.ds_factor_tiss)) + int(
                                   round(win_size[0] / self.ds_factor_tiss)),
                               int(round(col / self.ds_factor_tiss)):int(round(col / self.ds_factor_tiss)) + int(
                                   round(win_size[1] / self.ds_factor_tiss))]
                    if np.sum(win_tiss) > 0:
                        self.patch_coords.append([row, col])
                        self.patch_idx.append(idx)
                else:
                    self.patch_coords.append([row, col])
                idx += 1
        
        # generate array of zeros - will insert patch predictions later 
        self.zero_array = np.zeros([idx,self.mask_shape[0], self.mask_shape[1],9]) # 9 is the number of total output channels
    ####

    def load_batch(self, batch_coor, wsi_ext):
        """
        Loads a batch of images from provided coordinates.

        Args:
            batch_coor: list of coordinates in a batch
            wsi_ext   : file extension of the whole-slide image
        """

        batch = []
        win_size = self.input_shape
        if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
            win_size = np.int64(np.array(self.input_shape)/2)

        for coor in batch_coor:
            win = self.read_region((int(coor[1] * self.ds_factor), int(coor[0] * self.ds_factor)),
                                   self.proc_lvl, (win_size[0], win_size[1]), wsi_ext)
            if self.scan_resolution[0] > 0.35:  # it means image is scanned at 20X
                win = cv2.resize(win, (win.shape[1]*2, win.shape[0]*2), cv2.INTER_LINEAR) # cv.INTER_LINEAR is good for zooming
            batch.append(win)
        return batch
    ####

    def run_inference(self, tile, wsi_ext):
        """
        Run inference for extracted patches and apply post processing.
        Results are then assembled to the size of the original image.
        
        Args:
            tile: tile number index
            wsi_ext: file extension of the whole-slide image
        """

        pred_map_list = deque()
        mask_list = []
        type_list = []
        cent_list = []
        offset = (self.input_shape[0] - self.mask_shape[0]) / 2
        idx = 0
        batch_count = np.floor(len(self.patch_coords) / self.batch_size)

        if len(self.patch_coords) > 0:
            while len(self.patch_coords) > self.batch_size:
                sys.stdout.write("\rBatch (%d/%d) of Tile (%d/%d)" % (
                idx + 1, batch_count, tile + 1, self.nr_tiles_h * self.nr_tiles_w))
                sys.stdout.flush()
                idx += 1
                mini_batch_coor = self.patch_coords[:self.batch_size]
                mini_batch = self.load_batch(mini_batch_coor, wsi_ext)
                self.patch_coords = self.patch_coords[self.batch_size:]
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, self.batch_size, axis=0)
                pred_map_list.extend(mini_output)

            # Deal with the case when the number of patches is not divisisible by batch size
            if len(self.patch_coords) != 0:
                mini_batch = self.load_batch(self.patch_coords, wsi_ext)
                mini_output = self.predictor(mini_batch)[0]
                mini_output = np.split(mini_output, len(self.patch_coords), axis=0)
                pred_map_list.extend(mini_output)
        
            # Assemble back into full image
            output_patch_shape = np.squeeze(pred_map_list[0]).shape
            ch = 1 if len(output_patch_shape) == 2 else output_patch_shape[-1]

            pred_map = self.zero_array
            pred_map[np.array(self.patch_idx)] = np.squeeze(np.array(pred_map_list))
            pred_map = np.reshape(pred_map, (self.nr_step_h, self.nr_step_w) + pred_map.shape[1:])
            pred_map = np.transpose(pred_map, [0, 2, 1, 3, 4]) if ch != 1 else \
                            np.transpose(pred_map, [0, 2, 1, 3])
            pred_map = np.reshape(pred_map, (pred_map.shape[0] * pred_map.shape[1], 
                                            pred_map.shape[2] * pred_map.shape[3], ch))
            
            # crop back to original size
            if self.scan_resolution[0] > 0.35: # 20x
                pred_map = np.squeeze(pred_map[:self.tile_info[tile][3]*2,:self.tile_info[tile][2]*2])
            else:
                pred_map = np.squeeze(pred_map[:self.tile_info[tile][3],:self.tile_info[tile][2]]) 

            # post processing for a tile
            tile_coords = (self.tile_info[tile][0], self.tile_info[tile][1])
            mask_list, type_list, cent_list = proc_utils.process_instance_wsi(
                pred_map, self.nr_types, tile_coords, self.return_masks, offset=offset)
        else:
            mask_list = []
            type_list = []
            cent_list = []
        
        return mask_list, type_list, cent_list
    ####

    def process_wsi(self, filename):
        """
        Process an individual WSI. This function will:
        1) Load the OpenSlide WSI object
        2) Generate the tissue mask
        3) Get tile coordinate info
        4) Extract patches from foreground regions
        5) Run inference and return npz for each tile of 
           masks, type predictions and centroid locations
        """

        # Load the OpenSlide WSI object
        self.full_filename = self.input_dir + '/' + filename
        wsi_ext = self.full_filename.split('.')[-1]
        print(self.full_filename)
        self.load_wsi(wsi_ext)

        self.ds_factor = self.level_downsamples[self.proc_lvl]

        is_valid_tissue_level = True
        tissue_level = self.tiss_lvl
        if tissue_level < len(self.level_downsamples):  # if given tissue level exist
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_lvl]
        elif len(self.level_downsamples) > 1:
            tissue_level = len(self.level_downsamples) - 1  # to avoid tissue segmentation at level 0
            self.ds_factor_tiss = self.level_downsamples[tissue_level] / self.level_downsamples[self.proc_lvl]
        else:
            is_valid_tissue_level = False

        if self.tiss_seg & is_valid_tissue_level:
            # Generate tissue mask
            ds_img = self.read_region(
                (0, 0),
                tissue_level,
                (self.level_dimensions[tissue_level][1], self.level_dimensions[tissue_level][0]),
                wsi_ext
            )

            # downsampling factor if image is largest dimension of the image is greater than 5000 at given tissue level
            # to reduce tissue segmentation time
            proc_scale = 1 / np.ceil(np.max(ds_img.shape) / 5000)

            self.tissue = proc_utils.get_tissue_mask(ds_img, proc_scale)

        # Coordinate info for tile processing
        self.tile_coords()

        # Run inference tile by tile - if self.tiss_seg == True, only process tissue regions
        mask_list_all = []
        type_list_all = []
        cent_list_all = []
        for tile in range(len(self.tile_info)):

            self.extract_patches(tile)

            mask_list, type_list, cent_list = self.run_inference(tile, wsi_ext)

            # add tile predictions to overall prediction list
            mask_list_all.extend(mask_list)
            type_list_all.extend(type_list)
            cent_list_all.extend(cent_list)
            
            # uncomment below if you want to save results per tile

            # np.savez('%s/%s/%s_%s.npz' % (
            # self.output_dir, self.basename, self.basename, str(tile)),
            # mask=mask_list, type=type_list, centroid=cent_list
            # )

        if self.ds_factor != 1:
            cent_list = self.ds_factor * np.array(cent_list)
            cent_list = cent_list.tolist()
        np.savez('%s/%s/%s.npz' % (
            self.output_dir, self.basename, self.basename),
            mask=mask_list_all, type=type_list_all, centroid=cent_list_all
            )
    ####

    def load_model(self):
        """
        Loads the model and checkpoints according to the model stated in config.py
        """

        print('Loading Model...')
        model_path = self.model_path
        model_constructor = self.get_model()
        pred_config = PredictConfig(
            model        = model_constructor(self.nr_types, self.input_shape, self.mask_shape, self.input_norm),
            session_init = get_model_loader(model_path),
            input_names  = self.input_tensor_names,
            output_names = self.output_tensor_names)
        self.predictor = OfflinePredictor(pred_config)
    ####

    def load_filenames(self):
        """
        Get the list of all WSI files to process
        """
        self.file_list = glob.glob('%s/*' %self.input_dir)
        self.file_list.sort() # ensure same order
####
    
    def process_all_wsi(self):
        """
        Process each WSI one at a time and save results as npz file
        """

        if os.path.isdir(self.output_dir) == False:
            rm_n_mkdir(self.output_dir)

        for filename in self.file_list:
            filename = os.path.basename(filename)
            self.basename = os.path.splitext(filename)[0]
            # this will overwrite file is it was processed previously
            rm_n_mkdir(self.output_dir + '/' + self.basename)
            start_time_total = time.time()
            self.process_wsi(filename)
            end_time_total = time.time()
            print('. FINISHED. Time: ', time_it(start_time_total, end_time_total), 'secs')
        

#####
if __name__ == '__main__':
    args = docopt(__doc__, version='HoVer-Net Inference v1.0')
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['--gpu']

    # Raise exceptions for invalid / missing arguments
    if args['--model'] == None:
        raise Exception('A model path must be supplied as an argument with --model.')
    if args['--mode'] != 'roi' and args['--mode'] != 'wsi':
        raise Exception('Mode not recognised. Use either "roi" or "wsi"')
    if args['--input_dir'] == None:
        raise Exception('An input directory must be supplied as an argument with --input_dir.')
    if args['--input_dir'] == args['--output_dir']:
        raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')

    # Import libraries for WSI processing
    if args['--mode'] == 'wsi':
        import openslide as ops 
        try:
            import matlab
            from matlab import engine
        except:
            pass

    if args['--mode'] == 'roi':
        infer = InferROI()
        infer.load_params(args)
        infer.load_model() 
        infer.process()
    elif args['--mode'] == 'wsi': # currently saves results per tile
        infer = InferWSI()
        infer.load_params(args)
        infer.load_model() 
        infer.load_filenames()
        infer.process_all_wsi() 
