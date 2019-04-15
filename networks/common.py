# all building blocks to come here.

# downsampling layer comes here, or united.

import tensorflow as tf


def conv2d_with_batch_norm(input_, filter, kernel, stride=1, padding='valid',
                           dilation=1, use_bias=False):
    """Conv2d layer with batch normalisation."""
    out = tf.layers.conv2d(input_, filters=filter, kernel_size=kernel,
                           strides=stride, padding=padding,
                           dilation_rate=dilation, use_bias=use_bias)
    out = tf.layers.batch_normalization(out, momentum=1 - 0.1, epsilon=1e-5)
    return out
