

import importlib
import cv2
import numpy as np
import tensorflow as tf

#### 
class Config(object):
    def __init__(self, ):

        mode = 'hover'
        self.model_type = 'np_hv'

        self.type_classification = True # whether to predict the nuclear type
        # ! must use CoNSeP dataset, where nuclear type labels are available
        self.nr_types = 6  # denotes number of classes for nuclear type classification
        self.nr_classes = 2 # Nuclei Pixels vs Background

        #### Dynamically setting the config file into variable
        config_file = importlib.import_module('opt.hover') # np_hv, np_dist
        config_dict = config_file.__getattribute__(self.model_type)

        for variable, value in config_dict.items():
            self.__setattr__(variable, value)

        self.infer_input_shape = [256, 256]
        self.infer_mask_shape = [164, 164] 
        self.inf_batch_size = 25

        # number of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        self.input_norm  = True # normalize RGB to 0-1 range

        #### Info for running inference
        self.inf_model_path  = 'weights/hovernet.npz'

        # WSI processing
        self.tissue_inf = True
        self.inf_wsi_ext = '.svs'
        self.inf_wsi_dir = '/wsi_data_path/'
        self.proc_level = 2
        self.tiss_level = 3
        self.nr_tiles_h = 4
        self.nr_tiles_w = 4

        # ROI processing
        self.inf_imgs_ext = '.png'
        self.inf_data_dir = '/roi_data_path/'

        self.inf_output_dir = '/output_path/'

        # for inference during evalutaion mode i.e run by infer.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    def get_model(self):
        model_constructor = importlib.import_module('model.graph')
        model_constructor = model_constructor.Model_NP_HV  
        return model_constructor # NOTE return alias, not object
