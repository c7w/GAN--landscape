import IPython
import jittor as jt
import jittor.nn as nn
from functools import partial

from models.Transformer.SegFormer.OverlapPatchEmbed import OverlapPatchEmbed
from models.Transformer.SegFormer.SegFormerBlock import SegFormerBlock
from models.utils.init_weights import init_weights


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=(384, 512), patch_size=(16, 16), in_channels=3,
                 num_classes=29, embed_dims=(64, 128, 256, 512), num_heads=(1, 2, 4, 8),
                 mlp_ratios=(4, 4, 4, 4), qkv_bias=True, qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super(MixVisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=(7, 7), stride=patch_size[0],
                                              in_channels=in_channels, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4), patch_size=(3, 3), stride=2,
                                              in_channels=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8), patch_size=(3, 3), stride=2,
                                              in_channels=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 16, img_size[1] // 16), patch_size=(3, 3), stride=2,
                                              in_channels=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        # transformer encoder
        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        cur = 0
        self.block1 = nn.ModuleList([SegFormerBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([SegFormerBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([SegFormerBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([SegFormerBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(init_weights)

    def execute(self, x):
        B, C, H, W = x.shape

        outs = []  # Init return val array

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, block in enumerate(self.block1):
            x = block(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, block in enumerate(self.block2):
            x = block(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, block in enumerate(self.block3):
            x = block(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, block in enumerate(self.block4):
            x = block(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)

        return outs


class SegFormerB3(MixVisionTransformer):
    def __init__(self):
        super(SegFormerB3, self).__init__(
            patch_size=(2, 2), embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 4, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
