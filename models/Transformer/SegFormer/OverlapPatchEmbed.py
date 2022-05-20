import jittor as jt
import jittor.nn as nn

from models.utils.init_weights import init_weights


class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(7, 7), stride=4, in_channels=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]

        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[0] // 2), bias=True)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(init_weights)

    def execute(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)  # B, N, C

        return x, H, W


