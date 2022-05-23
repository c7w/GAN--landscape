import jittor.nn as nn
import math

def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.2)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

    elif isinstance(layer, nn.Conv2d):
        fan_out = layer.kernel_size[0] * layer.kernel_size[1] * layer.out_channels // layer.groups
        nn.init.trunc_normal_(layer.weight, std=math.sqrt(2.0 / fan_out))
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)

    elif isinstance(layer, nn.LayerNorm):
        nn.init.constant_(layer.bias, 0.0)
        nn.init.constant_(layer.weight, 1.0)

    else:
        pass
