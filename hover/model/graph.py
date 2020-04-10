import tensorflow as tf
from tensorpack.models import BatchNorm, BNReLU, Conv2D, MaxPooling, FixedUnPooling
from .utils import *

import sys
sys.path.append("..") # adds higher directory to python modules path.

####
def upsample2x(name, x):
    """
    Nearest neighbour up-sampling
    """
    return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
                data_format='channels_first')
####
def res_blk(name, l, ch, ksize, count, split=1, strides=1):
    ch_in = l.get_shape().as_list()
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block' + str(i)):  
                x = l if i == 0 else BNReLU('preact', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], split=split, 
                                strides=strides if i == 0 else 1, activation=BNReLU)
                x = Conv2D('conv3', x, ch[2], ksize[2], activation=tf.identity)
                if (strides != 1 or ch_in[1] != ch[2]) and i == 0:
                    l = Conv2D('convshortcut', l, ch[2], 1, strides=strides)
                l = l + x
        # end of each group need an extra activation
        l = BNReLU('bnlast',l)  
    return l
####
def dense_blk(name, l, ch, ksize, count, split=1, padding='valid'):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('blk/' + str(i)):
                x = BNReLU('preact_bna', l)
                x = Conv2D('conv1', x, ch[0], ksize[0], padding=padding, activation=BNReLU)
                x = Conv2D('conv2', x, ch[1], ksize[1], padding=padding, split=split)
                ##
                if padding == 'valid':
                    x_shape = x.get_shape().as_list()
                    l_shape = l.get_shape().as_list()
                    l = crop_op(l, (l_shape[2] - x_shape[2], 
                                    l_shape[3] - x_shape[3]))

                l = tf.concat([l, x], axis=1)
        l = BNReLU('blk_bna', l)
    return l
####
def encoder(i):
    """
    Pre-activated ResNet50 Encoder
    """

    d1 = Conv2D('conv0',  i, 64, 7, padding='same', strides=1, activation=BNReLU)
    d1 = res_blk('group0', d1, [ 64,  64,  256], [1, 3, 1], 3, strides=1)                       
    
    d2 = res_blk('group1', d1, [128, 128,  512], [1, 3, 1], 4, strides=2)

    d3 = res_blk('group2', d2, [256, 256, 1024], [1, 3, 1], 6, strides=2)

    d4 = res_blk('group3', d3, [512, 512, 2048], [1, 3, 1], 3, strides=2)
    
    d4 = Conv2D('conv_bot',  d4, 1024, 1, padding='same')
    return [d1, d2, d3, d4]
####
def decoder(name, i):
    pad = 'valid' # to prevent boundary artifacts
    with tf.variable_scope(name):
        with tf.variable_scope('u3'):
            u3 = upsample2x('rz', i[-1])
            u3_sum = tf.add_n([u3, i[-2]])

            u3 = Conv2D('conva', u3_sum, 256, 3, strides=1, padding=pad)   
            u3 = dense_blk('dense', u3, [128, 32], [1, 3], 8, split=4, padding=pad)
            u3 = Conv2D('convf', u3, 512, 1, strides=1)   
        ####
        with tf.variable_scope('u2'):          
            u2 = upsample2x('rz', u3)
            u2_sum = tf.add_n([u2, i[-3]])

            u2x = Conv2D('conva', u2_sum, 128, 3, strides=1, padding=pad)
            u2 = dense_blk('dense', u2x, [128, 32], [1, 3], 4, split=4, padding=pad)
            u2 = Conv2D('convf', u2, 256, 1, strides=1)   
        ####
        with tf.variable_scope('u1'):          
            u1 = upsample2x('rz', u2)
            u1_sum = tf.add_n([u1, i[-4]])

            u1 = Conv2D('conva', u1_sum, 64, 3, strides=1, padding='same')

    return [u3, u2x, u1]
####

class Model(ModelDesc):
    def __init__(self):
        super(Model, self).__init__()
        assert tf.test.is_gpu_available()

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None] + self.input_shape + [3], 'images'),
                InputDesc(tf.float32, [None] + self.mask_shape  + [None], 'truemap-coded')]
####

class Model_NP_HV(Model):
    def __init__(self, nr_types, input_shape, mask_shape, input_norm):
        self.nr_types = nr_types
        self.input_shape = input_shape
        self.mask_shape = mask_shape
        self.input_norm = input_norm 
        self.data_format = 'NCHW'

    def _build_graph(self, inputs):
        
        images, truemap_coded = inputs

        ####
        with argscope(Conv2D, activation=tf.identity, use_bias=False, # K.he initializer
                      W_init=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, BatchNorm], data_format=self.data_format):

            i = tf.transpose(images, [0, 3, 1, 2])
            i = i if not self.input_norm else i / 255.0

            ####
            d = encoder(i)
            d[0] = crop_op(d[0], (92, 92))
            d[1] = crop_op(d[1], (36, 36))

            ####
            np_feat = decoder('np', d)
            npx = BNReLU('preact_out_np', np_feat[-1])

            hv_feat = decoder('hv', d)
            hv = BNReLU('preact_out_hv', hv_feat[-1])

            tp_feat = decoder('tp', d)
            tp = BNReLU('preact_out_tp', tp_feat[-1])

            # Nuclei Type Pixels (TP)
            logi_class = Conv2D('conv_out_tp', tp, self.nr_types, 1, use_bias=True, activation=tf.identity)
            logi_class = tf.transpose(logi_class, [0, 2, 3, 1])
            soft_class = tf.nn.softmax(logi_class, axis=-1)

            #### Nuclei Pixels (NP)
            logi_np = Conv2D('conv_out_np', npx, 2, 1, use_bias=True, activation=tf.identity)
            logi_np = tf.transpose(logi_np, [0, 2, 3, 1])
            soft_np = tf.nn.softmax(logi_np, axis=-1)
            prob_np = tf.identity(soft_np[...,1], name='predmap-prob-np')
            prob_np = tf.expand_dims(prob_np, axis=-1)

            #### Horizontal-Vertival (HV)
            logi_hv = Conv2D('conv_out_hv', hv, 2, 1, use_bias=True, activation=tf.identity)
            logi_hv = tf.transpose(logi_hv, [0, 2, 3, 1])
            prob_hv = tf.identity(logi_hv, name='predmap-prob-hv')
            pred_hv = tf.identity(logi_hv, name='predmap-hv')
    
            # * channel ordering: type-map, segmentation map
            # encoded so that inference can extract all output at once
            predmap_coded = tf.concat([soft_class, prob_np, pred_hv], axis=-1, name='predmap-coded')

        return
####
