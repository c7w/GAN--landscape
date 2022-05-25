import IPython
import jittor as jt
import jittor.nn as nn
import numpy as np

from models.utils.init_weights import init_weights
from models.utils.utils import start_grad, stop_grad, weights_init_normal


class ConvGlobDiscriminator(nn.Module):

    def __init__(self, in_channels=3):
        super(ConvGlobDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns down-sampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.tanh = nn.Tanh()

        self.d1 = nn.Sequential(discriminator_block(32, 64, normalization=False))  # (B, 64, 192, 256)
        self.score1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=2, stride=2, padding=0),  # (B, 16, 96, 128)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 16, 48, 64)
            nn.LeakyReLU(scale=0.2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0),  # (B, 1, 24, 32)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 1, 12, 16)
            nn.Tanh()  # Normalize between -1 and 1
        )

        self.d2 = nn.Sequential(discriminator_block(64, 128))  # (B, 128, 96, 128)
        self.score2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=2, stride=2, padding=0),  # (B, 32, 48, 64)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 32, 24, 32)
            nn.LeakyReLU(scale=0.2),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=2, stride=2, padding=0),  # (B, 1, 12, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 1, 6, 8)
            nn.Tanh()  # Normalize between -1 and 1
        )

        self.d3 = nn.Sequential(discriminator_block(128, 256))
        self.score3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=2, stride=2, padding=0),  # (B, 64, 24, 32)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 64, 12, 16)
            nn.LeakyReLU(scale=0.2),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=2, stride=2, padding=0),  # (B, 1, 6, 8)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 1, 3, 4)
            nn.Tanh()  # Normalize between -1 and 1
        )

        self.d4 = nn.Sequential(discriminator_block(256, 512))
        self.score4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=2, stride=2, padding=0),  # (B, 128, 12, 16)
            nn.MaxPool2d(kernel_size=2, stride=2),  # (B, 128, 6, 8)
            nn.LeakyReLU(scale=0.2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=2, stride=2, padding=0),  # (B, 1, 3, 4)
        )
        self.score42 = nn.Sequential(
            nn.Linear(in_features=12, out_features=1),
            nn.Tanh()  # Normalize between -1 and 1
        )

        for m in self.modules():
            m.apply(init_weights)

    def execute(self, img, label):

        # First, convert label to label map
        label = label.numpy()
        label_map = np.concatenate([label == k for k in range(29)], axis=1).astype(np.float32)
        label_map = jt.array(label_map)

        x = jt.concat((img, label_map), dim=1)

        r1 = self.d1(x)  # (B, 64, 192, 256)
        score1 = self.score1(r1).sum(dims=(1, 2, 3)) / (12 * 16)  # (B, 1, 12, 16)

        r2 = self.d2(r1)  # (B, 128, 96, 128)
        score2 = self.score2(r2).sum(dims=(1, 2, 3)) / (6 * 8)  # (B, 1, 6, 8)

        r3 = self.d3(r2)  # (B, 256, 48, 64)
        score3 = self.score3(r3).sum(dims=(1, 2, 3)) / (3 * 4)  # (B, 1, 3, 4)

        r4 = self.d4(r3)  # (B, 512, 24, 32)
        score4 = self.score4(r4)  # (B, 1, 3, 4)
        score4 = score4.reshape(score4.shape[0], -1)  # (B, 12)
        score42 = self.score42(score4)  # (B, 1)

        # <-- Local --- Penalty --- Global -->
        weights = (2.0, 1.0, 0.5, 0.5)  # TODO: Try different weights?
        scores = (score1, score2, score3, score42)

        # Compute loss
        # Use negative value, because MaxPooling is used to find places with most 违和感.
        # More 违和感 the generated image has, the closer the loss should be to -1.0.
        loss = 0.0

        for i, w in enumerate(weights):
            loss += - w * scores[i]

        return loss / sum(weights)