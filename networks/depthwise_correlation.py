import tensorflow as tf

from .common import conv2d_with_batch_norm


def adjust(input_, scope):
    with tf.variable_scope('Adjust/' + scope):
        out = conv2d_with_batch_norm(input_, 256, 3)
        out = tf.nn.relu(out)
    return out


def conv2d_dw_group(input_, kernel):
    batch_size, i_h, i_w, i_c = input_.shape
    _, k_h, k_w, k_c = kernel.shape
    input_ = tf.transpose(input_, [1, 2, 0, 3])
    kernel = tf.transpose(kernel, [1, 2, 0, 3])
    input_ = tf.reshape(input_,
                        [1, i_h, i_w, batch_size * i_c])
    kernel = tf.reshape(kernel,
                        [k_h, k_w, batch_size * k_c, 1])
    out = tf.nn.depthwise_conv2d(input_, kernel, [1, 1, 1, 1], 'VALID')
    out = tf.reshape(out,
                     [batch_size, out.shape[1], out.shape[2], k_c])
    return out


def depthwise_corr(search, exemplar):
    adjusted_exemplar = adjust(exemplar, 'Exemplar')
    adjusted_search = adjust(search, 'Search')

    correlated_features = conv2d_dw_group(adjusted_search, adjusted_exemplar)
    return correlated_features
