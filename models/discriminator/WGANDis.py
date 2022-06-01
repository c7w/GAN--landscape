from jittor import nn
import numpy as np

img_shape = (3, 1024, 768)

class WGANDis(nn.Module):

    def __init__(self):
        super(WGANDis, self).__init__()
        self.model = nn.Sequential(nn.Linear(int(np.prod(img_shape)), 512),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(512, 256),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(256, 1),
                                   )

    def execute(self, img):
        img_flat = img.reshape((img.shape[0], (- 1)))
        validity = self.model(img_flat)
        return validity
