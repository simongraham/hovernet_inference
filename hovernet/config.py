

import importlib

#### 
class Config(object):
    def __init__(self, ):

        mode = 'hover'
        self.model_type = 'np_hv'

        self.type_classification = True # whether to predict the nuclear type
        self.nr_types = 6  # denotes number of classes (including BG) for nuclear type classification
        self.nr_classes = 2 # Nuclei Pixels vs Background

        self.infer_input_shape = [256, 256]
        self.infer_mask_shape = [164, 164] 
        self.inf_batch_size = 25

        # number of processes for parallel processing input
        self.nr_procs_train = 8 
        self.nr_procs_valid = 4 

        self.input_norm  = True # normalize RGB to 0-1 range

        #### Info for running inference
        self.inf_model_path  = '/home/jevjev/hovernet.npz'

        # WSI processing
        self.tissue_inf = True
        self.inf_wsi_ext = '.svs'
        self.inf_wsi_dir = '/path_to_wsis/'
        self.proc_level = 0
        self.tiss_level = 4
        self.nr_tiles_h = 5
        self.nr_tiles_w = 5

        # ROI processing
        self.inf_imgs_ext = '.png'
        self.inf_data_dir = '/home/jevjev/test_roi/'

        self.inf_output_dir = '/home/jevjev/roi_output/'

        if self.inf_wsi_dir == self.inf_data_dir or self.inf_output_dir == self.inf_data_dir:
            raise Exception('Input and output directories should not be the same- otherwise input directory will be overwritten.')

        # for inference during evalutaion mode i.e run by run_inference.py
        self.eval_inf_input_tensor_names = ['images']
        self.eval_inf_output_tensor_names = ['predmap-coded']
        # for inference during training mode i.e run by trainer.py
        self.train_inf_output_tensor_names = ['predmap-coded', 'truemap-coded']

    def get_model(self):
        model_constructor = importlib.import_module('model.graph')
        model_constructor = model_constructor.Model_NP_HV  
        return model_constructor # NOTE return alias, not object
