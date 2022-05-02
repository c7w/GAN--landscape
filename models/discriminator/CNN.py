import jittor as jt
import jittor.nn as nn
from models import weights_init_normal

class CNNDiscriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(CNNDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns down-sampling layers of each discriminator block"""
            layers = [nn.Conv(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        self.model = nn.Sequential(*discriminator_block((in_channels * 2), 64, normalization=False), *discriminator_block(64, 128), *discriminator_block(128, 256), *discriminator_block(256, 512, stride=1), nn.Conv(512, 1, 4, padding=1, bias=False))

        for m in self.modules():
            weights_init_normal(m)

    def execute(self, input):
        return self.model(input)
