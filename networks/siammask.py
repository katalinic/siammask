import tensorflow as tf

from .common import conv2d_with_batch_norm
from .resnet import ResNet50
from .depthwise_correlation import depthwise_corr


def head(features, out_channels):
    with tf.variable_scope('Head'):
        out = conv2d_with_batch_norm(features, 256, 1)
        out = tf.nn.relu(out)
        out = tf.layers.conv2d(out, filters=out_channels, kernel_size=1,
                               use_bias=True)
    return out


def branch(search_features, exemplar_features, output_size, scope):
    with tf.variable_scope(scope):
        correlated_features = depthwise_corr(
            search_features, exemplar_features)

        output = head(correlated_features, output_size)

    return output


def bboxes_with_scores(search, exemplar, num_anchors):
    search_features = ResNet50(search, downsample=True)
    exemplar_features = ResNet50(exemplar, downsample=True)

    scores = branch(search_features, exemplar_features, 2 * num_anchors,
                    'Score')
    boxes = branch(search_features, exemplar_features, 4 * num_anchors,
                   'BBox')

    return scores, boxes
