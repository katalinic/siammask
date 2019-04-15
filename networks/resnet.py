from functools import partial

import tensorflow as tf

from .common import conv2d_with_batch_norm


def residual_block(input_, filter, stride=1, padding='SAME', dilation=1,
                   downsample=None, scope=''):
    """Block of conv2d layers to be repeated."""
    out = input_
    residual = input_

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        out = conv2d_with_batch_norm(out, filter, 1)
        out = tf.nn.relu(out)
        out = conv2d_with_batch_norm(out, filter, 3, stride, padding, dilation)
        out = tf.nn.relu(out)
        out = conv2d_with_batch_norm(out, filter * 4, 1)

        if downsample is not None:
            with tf.variable_scope('Downsample'):
                residual = downsample(input_)

    msg = 'Output and residual shapes do not match.'
    assert out.shape[1:] == residual.shape[1:], msg

    out += residual
    out = tf.nn.relu(out)
    return out


def resnet_layer(input_, num_blocks, filter, stride=1, dilation=1, scope=''):
    """A resnet layer composed of repeated residual blocks."""
    kernel = 1 if (stride == 1 and dilation == 1) else 3
    padding = 'SAME' if dilation > 1 else 'VALID'
    dd = 1
    downsample = partial(conv2d_with_batch_norm, filter=filter * 4,
                         kernel=kernel, stride=stride, padding=padding,
                         dilation=dd)
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        padding = 'VALID' if stride == 2 else 'SAME'
        out = residual_block(input_, filter, stride, padding, dd,
                             downsample, 'Block_0')
        for i in range(1, num_blocks):
            if scope == 'Layer4':
                out = residual_block(out, filter, stride, padding, dilation,
                                     scope='Block_{}'.format(i))
            else:
                out = residual_block(out, filter, scope='Block_{}'.format(i))
    return out


def _downsample(input_, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output = conv2d_with_batch_norm(input_, 256, 1)
        if output.shape[2] < 20:
            l, r = 4, -4
            output = output[:, l:r, l:r, :]
        return output


def ResNet50(input_, downsample=False):
    """Modified ResNet 50."""
    with tf.variable_scope('Layer1', reuse=tf.AUTO_REUSE):
        conv1 = conv2d_with_batch_norm(input_, 64, 7, 2, 'VALID', 1)
        conv1 = tf.nn.relu(conv1)
    conv2 = tf.layers.max_pooling2d(conv1, 3, 2, padding='SAME')
    conv2 = resnet_layer(conv2, 3, 64, 1, scope='Layer2')
    conv3 = resnet_layer(conv2, 4, 128, 2, scope='Layer3')
    conv4 = resnet_layer(conv3, 6, 256, 1, 2, scope='Layer4')
    if downsample:
        conv4 = _downsample(conv4, 'Downsample')
        return conv4
