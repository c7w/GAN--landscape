import numpy as np
import jittor as jt
import jittor.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return DropPath.drop_path(x, self.drop_prob, self.is_training())

    @classmethod
    def drop_path(cls, x, drop_prob=0.0, training=False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob

        shape = [x.shape[0], *[1 for i in range(len(x.shape()) - 1)]]
        random_tensor = jt.rand(shape, dtype=x.dtype) + keep_prob
        np.floor(random_tensor.data, out=random_tensor.data)

        return x / keep_prob * random_tensor
