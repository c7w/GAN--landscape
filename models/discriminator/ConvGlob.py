import IPython
import jittor as jt
import jittor.nn as nn
import numpy as np

from models.utils.init_weights import init_weights
from models.utils.utils import start_grad, stop_grad, weights_init_normal


class ConvGlobDiscriminator(nn.Module):
    """
    TODO: 'Memorize' strategies is strongly needed, maybe add the L1 Loss back...
    TODO: RGB Image channels and Semantic Label channels should not be viewed as duals, but now they are just\
     concatenated together.
    TODO: Maybe change discriminator to Attention Network to penalize repeating?
    """

    def __init__(self, in_channels=3):
        super(ConvGlobDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalization=True):
            """Returns down-sampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers


        self.d1 = nn.Sequential(discriminator_block(6, 64, normalization=False))  # (B, 64, 192, 256)
        self.score1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (B, 16, 96, 128)
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=2, padding=1),  # (B, 16, 96, 128)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 16, 48, 64)
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),  # (B, 16, 48, 64)
            nn.BatchNorm2d(16, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1),  # (B, 1, 24, 32)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 1, 12, 16)
        )

        self.d2 = nn.Sequential(discriminator_block(64, 128))  # (B, 128, 96, 128)
        self.score2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (B, 128, 96, 128)
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1),  # (B, 32, 48, 64)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 32, 24, 32)
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # (B, 32, 24, 32)
            nn.BatchNorm2d(32, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),  # (B, 1, 12, 16)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 1, 6, 8)
        )

        self.d3 = nn.Sequential(discriminator_block(128, 256))  # (B, 256, 48, 64)
        self.score3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (B, 256, 48, 64)
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),  # (B, 64, 24, 32)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 64, 12, 16)
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # (B, 64, 12, 16)
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),  # (B, 1, 6, 8)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 1, 3, 4)
        )

        self.d4 = nn.Sequential(discriminator_block(256, 512))
        self.score4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # (B, 512, 24, 32)
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1),  # (B, 128, 12, 16)
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),  # (B, 128, 6, 8)
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # (B, 128, 6, 8)
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(scale=0.2),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=2, padding=1),  # (B, 1, 3, 4)
        )
        self.score42 = nn.Sequential(
            nn.Linear(in_features=12, out_features=1),
        )

        self.weight = nn.Sequential(
            nn.Linear(in_features=4, out_features=1),
            nn.Tanh()  # Normalize between -1 and 1
        )

        for m in self.modules():
            m.apply(init_weights)

    def execute(self, input):

        # First, convert label to label map
        # label = label.numpy()
        # label_map = np.concatenate([label == k for k in range(29)], axis=1).astype(np.float32)
        # label_map = jt.array(label_map)

        # x = jt.concat((img, label_map), dim=1)

        B = input.shape[0]

        r1 = self.d1(input)  # (B, 64, 192, 256)
        score1 = self.score1(r1).sum(dims=(1, 2, 3)) / (12 * 16)  # (B, 1, 12, 16)

        r2 = self.d2(r1)  # (B, 128, 96, 128)
        score2 = self.score2(r2).sum(dims=(1, 2, 3)) / (6 * 8)  # (B, 1, 6, 8)

        r3 = self.d3(r2)  # (B, 256, 48, 64)
        score3 = self.score3(r3).sum(dims=(1, 2, 3)) / (3 * 4)  # (B, 1, 3, 4)

        r4 = self.d4(r3)  # (B, 512, 24, 32)
        score4 = self.score4(r4)  # (B, 1, 3, 4)
        score4 = score4.reshape(score4.shape[0], -1)  # (B, 12)
        score42 = self.score42(score4)  # (B, 1)

        score = self.weight(
            jt.concat((score1.reshape(B, 1), score2.reshape(B, 1), score3.reshape(B, 1), score42.reshape(B, 1)), dim=1)
        ).sum() / B

        # # <-- Local --- Penalty --- Global -->
        # weights = (2.0, 1.0, 0.5, 0.5)  # TODO: Try different weights?
        # scores = (score1, score2, score3, score42)
        #
        # # Compute loss
        # # Use negative value, because MaxPooling is used to find places with most 违和感.
        # # More 违和感 the generated image has, the closer the loss should be to -1.0.
        # loss = 0.0
        #
        # for i, w in enumerate(weights):
        #     loss += - w * scores[i]

        return score
