# SiamMask.

TensorFlow port of [SiamMask](https://arxiv.org/abs/1812.05050). Mask branch is yet to be added.

Pre-trained models were ported from PyTorch to TensorFlow and can be downloaded [here](https://drive.google.com/open?id=1YQNXJgezEciQbN0Mwyta50UzVamNjswe). Many functions for tracking and processing outside the trained models were
taken directly (at most with minor modification) from the [SiamMask Github](https://github.com/foolwood/SiamMask).

The demo (after downloading the model into the `saved_model` folder) can be run as:

`$ python3 demo.py`
