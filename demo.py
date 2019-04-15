import cv2
import numpy as np
import tensorflow as tf

from config import Config
from networks.siammask import bboxes_with_scores
from saved_model.json_model_utils import load_json
from tracking import (tracking_init, preprocess, update_bounding_box,
                      _example_wh_to_size, _search_wh_to_size, _create_polygon)

DATA_LOCATION = 'datasets/tennis/'
MODEL_PATH = 'saved_model/SiamMask_DAVIS.json'


def get_image_filenames(dir):
    import os
    import sys
    return sorted([sys.path[0] + '/' + dir + s for s in os.listdir(dir)
                  if s.endswith(('.jpg', '.png'))])


def run():

    image_filenames = get_image_filenames(DATA_LOCATION)
    images = [cv2.imread(filename) for filename in image_filenames]
    cfg = Config()

    search_pl = tf.placeholder(tf.float32,
                               [1, cfg.search_size, cfg.search_size, 3])
    example_pl = tf.placeholder(tf.float32,
                                [1, cfg.exemplar_size, cfg.exemplar_size, 3])
    scores, bboxes = bboxes_with_scores(
        search_pl, example_pl, cfg.anchor.num_anchors)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    load_json(sess, vars, MODEL_PATH)

    anchors, window = tracking_init(cfg)

    # Hardcoded starting polygon for demo; top left corner as location.
    target = [310, 120, 160, 250]
    x, y, w, h = target
    target_pos = np.array([x + w / 2, y + h / 2])
    target_size = np.array([w, h])

    example_size_ = _example_wh_to_size(target_size, cfg)

    processed_example = preprocess(
        images[0], target_pos, example_size_, cfg.exemplar_size)

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)

    for image in images[1:]:
        search_size_, search_scale = _search_wh_to_size(
            target_size, cfg)

        processed_search = preprocess(
            image, target_pos, round(search_size_), cfg.search_size)

        scores_, bboxes_ = sess.run(
            [scores, bboxes],
            feed_dict={search_pl: processed_search,
                       example_pl: processed_example})

        target_pos, target_size = update_bounding_box(
            image, scores_, bboxes_, anchors, window, target_pos, target_size,
            search_scale, cfg)

        polygon = _create_polygon(target_pos, target_size)

        cv2.polylines(image, [polygon], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', image)

        key = cv2.waitKey(1)
        if key > 0:
            break


if __name__ == '__main__':
    run()
