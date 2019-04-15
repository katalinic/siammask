import cv2
import numpy as np

from anchors import init_anchors


def tracking_init(cfg):
    anchors = init_anchors(cfg)
    window = np.outer(np.hanning(cfg.score_size), np.hanning(cfg.score_size))
    window = np.tile(window.flatten(), cfg.anchor.num_anchors)
    return anchors, window


def preprocess(image, center, crop_size, output_size):
    image_size = image.shape
    c = (crop_size + 1) / 2
    context_xmin = round(center[0] - c)
    context_xmax = context_xmin + crop_size - 1
    context_ymin = round(center[1] - c)
    context_ymax = context_ymin + crop_size - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - image_size[1] + 1))
    bottom_pad = int(max(0., context_ymax - image_size[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    avg_chans = np.mean(image, (0, 1))

    r, c, k = image.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        padded_im = np.zeros(
            (r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
        padded_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = image
        if top_pad:
            padded_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            padded_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            padded_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            padded_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = padded_im[int(context_ymin): int(context_ymax + 1),
                                      int(context_xmin): int(context_xmax + 1),
                                      :]
    else:
        im_patch_original = image[int(context_ymin): int(context_ymax + 1),
                                  int(context_xmin): int(context_xmax + 1),
                                  :]
    if not np.array_equal(output_size, crop_size):
        im_patch = cv2.resize(im_patch_original, (output_size, output_size))
    else:
        im_patch = im_patch_original

    im_patch = np.expand_dims(im_patch, 0)
    return im_patch


def softmax(x):
    adjusted_x = x - np.amax(x, axis=-1, keepdims=-1)
    numerator = np.exp(adjusted_x)
    denominator = np.sum(numerator, axis=-1, keepdims=-1)
    return numerator/denominator


def update_bounding_box(image, scores, bboxes, anchors, window,
                        target_pos, target_size, search_scale, cfg):
    bboxes = np.transpose(bboxes, [3, 1, 2, 0]).reshape(4, -1)
    scores = softmax(np.transpose(scores, [3, 1, 2, 0]).reshape(2, -1).T)[:, 1]

    bboxes[0, :] = bboxes[0, :] * anchors[:, 2] + anchors[:, 0]
    bboxes[1, :] = bboxes[1, :] * anchors[:, 3] + anchors[:, 1]
    bboxes[2, :] = np.exp(bboxes[2, :]) * anchors[:, 2]
    bboxes[3, :] = np.exp(bboxes[3, :]) * anchors[:, 3]

    def change(r):
        return np.maximum(r, 1. / r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    target_sz_in_crop = target_size * search_scale

    s_c = change(
        sz(bboxes[2, :], bboxes[3, :]) / (sz_wh(target_sz_in_crop)))
    r_c = change(
        (target_sz_in_crop[0] / target_sz_in_crop[1])
        / (bboxes[2, :] / bboxes[3, :]))

    penalty = np.exp(-(r_c * s_c - 1) * cfg.penalty_k)
    pscore = penalty * scores

    # cos window (motion model)
    pscore = (pscore * (1 - cfg.window_influence)
              + window * cfg.window_influence)
    best_pscore_id = np.argmax(pscore)

    pred_in_crop = bboxes[:, best_pscore_id] / search_scale
    lr = penalty[best_pscore_id] * scores[best_pscore_id] * cfg.lr
    # print(lr, pred_in_crop)
    res_x = pred_in_crop[0] + target_pos[0]
    res_y = pred_in_crop[1] + target_pos[1]

    res_w = target_size[0] * (1 - lr) + pred_in_crop[2] * lr
    res_h = target_size[1] * (1 - lr) + pred_in_crop[3] * lr

    target_pos = np.array([res_x, res_y])
    target_size = np.array([res_w, res_h])

    h, w, _ = image.shape if len(image.shape) == 3 else image[0].shape

    target_pos[0] = max(0, min(w, target_pos[0]))
    target_pos[1] = max(0, min(h, target_pos[1]))
    target_size[0] = max(10, min(w, target_size[0]))
    target_size[1] = max(10, min(h, target_size[1]))

    return target_pos, target_size


def _example_wh_to_size(target_size, cfg):
    w, h = target_size
    target_sz = np.array([w, h])

    wc_z = target_sz[0] + cfg.context_amount * sum(target_sz)
    hc_z = target_sz[1] + cfg.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    return s_z


def _search_wh_to_size(target_size, cfg):
    w, h = target_size
    target_sz = np.array([w, h])

    wc_x = target_sz[0] + cfg.context_amount * sum(target_sz)
    hc_x = target_sz[1] + cfg.context_amount * sum(target_sz)
    s_x = np.sqrt(wc_x * hc_x)
    scale_x = cfg.exemplar_size / s_x
    d_search = (cfg.search_size - cfg.exemplar_size) / 2
    pad = d_search / scale_x
    s_x = s_x + 2 * pad

    return s_x, scale_x


def _create_polygon(loc, size):
    loc = np.array([[loc[0] - size[0] // 2, loc[1] - size[1] // 2],
                    [loc[0] - size[0] // 2, loc[1] + size[1] // 2],
                    [loc[0] + size[0] // 2, loc[1] + size[1] // 2],
                    [loc[0] + size[0] // 2, loc[1] - size[1] // 2]],
                   dtype=np.int32)
    loc = loc.reshape(-1, 1, 2)
    return loc
