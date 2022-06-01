from jittor import nn
import numpy as np

img_shape = (3, 1024, 768)

class W_GAN_GP_Generator(nn.Module):

    def __init__(self):
        super(W_GAN_GP_Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
            """
            latent_dim(隐藏层 64)不断升维, img_shape = (hannels, img_size, img_size)
            """
        self.model = nn.Sequential(*block(64, 128, normalize=False), \
            *block(128, 256), *block(256, 512), *block(512, 1024), \
                nn.Linear(1024, int(np.prod(img_shape))), nn.Tanh())

    def execute(self, z):
        img = self.model(z)
        img = img.view((img.shape[0], *img_shape))
        # view 重构了张量的形状，类似 resize
        return img
