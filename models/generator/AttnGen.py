import IPython
import jittor as jt
import jittor.nn as nn
import numpy as np


class AttnGen(nn.Module):
    def __init__(self):
        super(AttnGen, self).__init__()
        self.fc = nn.Linear(29, 3)

    def execute(self, label):
        label = label[:, 0, :, :][:, np.newaxis, :, :]

        # First, convert label (384x512) to label map (29x384x512)
        label_map = jt.concat([jt.int8(label == k) for k in range(29)], dim=1)
        IPython.embed()
        # Then, divide into patches and compute attention maps

        return self.fc(label_map)
        # return jt.array(np.zeros(shape=(3, 384, 512)))
