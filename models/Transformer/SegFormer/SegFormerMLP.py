import math

import jittor as jt
import jittor.nn as nn
from models.utils.init_weights import init_weights


class SegFormerMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, active_function=nn.GELU, drop=0.1):
        super(SegFormerMLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = nn.Conv2d(in_channels=hidden_features,
                              out_channels=hidden_features,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True,
                              groups=hidden_features)
        self.active_function = active_function()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(init_weights)

    def execute(self, x: jt.Var, H: int, W: int):
        x = self.fc1(x)

        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.active_function(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


