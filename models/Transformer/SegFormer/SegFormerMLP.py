import jittor as jt
import jittor.nn as nn

class SegFormerMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, active_function=nn.GELU, drop=0.1):
        super().__init__()
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

        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.2)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        elif isinstance(layer, nn.Conv2d):



        elif isinstance(layer, nn.LayerNorm):
            nn.init.constant_(layer.bias, 0.0)
            nn.init.constant_(layer.weight, 1.0)

        else:
            pass