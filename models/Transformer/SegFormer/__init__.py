import jittor as jt
import jittor.nn as nn

class SegFormer(nn.Module):
    """
    This class is the implementation of the SegFormer model introduced in
    https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html

    It takes an input of shape (B, 3, H, W) and returns an output of shape (B, Nc, H, W)
    """
    def __init__(self, height=384, width=512, num_classes=29, *args, **kw):
        super().__init__(*args, **kw)


    def execute(self, x):
        pass

