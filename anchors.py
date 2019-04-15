import math

import numpy as np


def generate_anchors(cfg):
    anchors = np.zeros((cfg.num_anchors, 4), dtype=np.float32)

    size = cfg.stride * cfg.stride
    count = 0
    for r in cfg.ratios:

        ws = int(math.sqrt(size * 1. / r))
        hs = int(ws * r)

        for s in cfg.scales:

            w = ws * s
            h = hs * s
            anchors[count] = 0.5 * np.array([-w, -h, w, h])
            count += 1

    return anchors


def prepare_anchors(anchor, cfg):
    x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
    anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)

    total_stride = cfg.stride
    anchor_num = anchor.shape[0]

    anchor = np.tile(anchor, cfg.score_size * cfg.score_size).reshape((-1, 4))
    b = - (cfg.score_size // 2) * total_stride
    xx, yy = np.meshgrid([b + total_stride * dx
                          for dx in range(cfg.score_size)],
                         [b + total_stride * dy
                          for dy in range(cfg.score_size)])
    xx, yy = (np.tile(xx.flatten(), (anchor_num, 1)).flatten(),
              np.tile(yy.flatten(), (anchor_num, 1)).flatten())
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


def init_anchors(cfg):
    anchors = generate_anchors(cfg.anchor)
    anchors = prepare_anchors(anchors, cfg)
    return anchors
