class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Config:
    search_size = 255
    exemplar_size = 127
    base_size = 8
    stride = 8
    score_size = (search_size - exemplar_size) // stride + base_size + 1

    penalty_k = 0.04
    window_influence = 0.4
    lr = 1.0
    windowing = 'cosine'
    context_amount = 0.5

    # Anchors.
    anchor = AttrDict(
        {
            'stride': 8,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8],
            'num_anchors': 5
        }
    )
