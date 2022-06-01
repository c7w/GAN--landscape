import IPython
import jittor as jt
import jittor.nn as nn
import numpy as np

from models.Transformer.SegFormer.MixVisionTransformer import SegFormerSmall


class AttnGen(nn.Module):
    def __init__(self):
        super(AttnGen, self).__init__()
        self.transformer = SegFormerSmall()
        self.header = AttnGenHeader()

    def execute(self, label):
        # First, convert label (384x512) to label map (29x384x512)
        label = label.numpy()
        label_map = np.concatenate([label == k for k in range(29)], axis=1).astype(np.float32)
        label_map = jt.array(label_map)

        # Then, divide into patches and compute attention maps
        features = self.transformer(label_map)
        result = self.header(features)
        return result


class AttnGenHeader(nn.Module):
    def __init__(self):
        self.in_channels = (64, 128, 320, 512)
        embedding_dim = 512
        hidden_dim = 768

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        self.linear_c1 = nn.Linear(c1_in_channels, embedding_dim)
        self.linear_c2 = nn.Linear(c2_in_channels, embedding_dim)
        self.linear_c3 = nn.Linear(c3_in_channels, embedding_dim)
        self.linear_c4 = nn.Linear(c4_in_channels, embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def execute(self, features):
        # x = self._transform_inputs(inputs)  # len=4, 1/4, 1/8, 1/16, 1/32
        c1, c2, c3, c4 = features

        ############## MLP decoder on C1-C4 ###########
        B, C, H, W = c4.shape

        _c4 = c4.reshape(B, c4.shape[1], -1).permute(0, 2, 1)
        _c4 = self.linear_c4(_c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = jt.nn.interpolate(_c4, size=(384, 512))

        _c3 = c3.reshape(B, c3.shape[1], -1).permute(0, 2, 1)
        _c3 = self.linear_c3(_c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = jt.nn.interpolate(_c3, size=(384, 512))

        _c2 = c2.reshape(B, c2.shape[1], -1).permute(0, 2, 1)
        _c2 = self.linear_c2(_c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = jt.nn.interpolate(_c2, size=(384, 512))

        _c1 = c1.reshape(B, c1.shape[1], -1).permute(0, 2, 1)
        _c1 = self.linear_c1(_c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])
        _c1 = jt.nn.interpolate(_c1, size=(384, 512))

        _c = self.linear_fuse(jt.concat([_c4, _c3, _c2, _c1], dim=1))
        return _c

