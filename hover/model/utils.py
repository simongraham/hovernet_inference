import numpy as np

import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from matplotlib import cm


def resize_op(
    x,
    height_factor=None,
    width_factor=None,
    size=None,
    interp="bicubic",
    data_format="channels_last",
):
    """
    Resize by a factor if `size=None` else resize to `size`
    """
    original_shape = x.get_shape().as_list()
    if size is not None:
        if data_format == "channels_first":
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, size)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape(
                (
                    None,
                    original_shape[1] if original_shape[1] is not None else None,
                    size[0],
                    size[1],
                )
            )
        else:
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, size)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, size)
            else:
                x = tf.image.resize_nearest_neighbor(x, size)
            x.set_shape(
                (
                    None,
                    size[0],
                    size[1],
                    original_shape[3] if original_shape[3] is not None else None,
                )
            )
    else:
        if data_format == "channels_first":
            new_shape = tf.cast(tf.shape(x)[2:], tf.float32)
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype("float32")
            )
            new_shape = tf.cast(new_shape, tf.int32)
            x = tf.transpose(x, [0, 2, 3, 1])
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x = tf.transpose(x, [0, 3, 1, 2])
            x.set_shape(
                (
                    None,
                    original_shape[1] if original_shape[1] is not None else None,
                    int(original_shape[2] * height_factor)
                    if original_shape[2] is not None
                    else None,
                    int(original_shape[3] * width_factor)
                    if original_shape[3] is not None
                    else None,
                )
            )
        else:
            original_shape = x.get_shape().as_list()
            new_shape = tf.cast(tf.shape(x)[1:3], tf.float32)
            new_shape *= tf.constant(
                np.array([height_factor, width_factor]).astype("float32")
            )
            new_shape = tf.cast(new_shape, tf.int32)
            if interp == "bicubic":
                x = tf.image.resize_bicubic(x, new_shape)
            elif interp == "bilinear":
                x = tf.image.resize_bilinear(x, new_shape)
            else:
                x = tf.image.resize_nearest_neighbor(x, new_shape)
            x.set_shape(
                (
                    None,
                    int(original_shape[1] * height_factor)
                    if original_shape[1] is not None
                    else None,
                    int(original_shape[2] * width_factor)
                    if original_shape[2] is not None
                    else None,
                    original_shape[3] if original_shape[3] is not None else None,
                )
            )
    return x


def crop_op(x, cropping, data_format="channels_first"):
    """
    Center crop image
    Args:
        cropping is the substracted portion
    """
    crop_t = cropping[0] // 2
    crop_b = cropping[0] - crop_t
    crop_l = cropping[1] // 2
    crop_r = cropping[1] - crop_l
    if data_format == "channels_first":
        x = x[:, :, crop_t:-crop_b, crop_l:-crop_r]
    else:
        x = x[:, crop_t:-crop_b, crop_l:-crop_r]
    return x
