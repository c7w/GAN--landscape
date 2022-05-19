import jittor as jt
import jittor.nn as nn

from models.Transformer.SegFormer.SegFormerAttention import SegFormerAttention


class SegFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, proj_drop=0.1, attn_drop=0.1, drop_path=0.0, activation_func=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.0):
        super(SegFormerBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SegFormerAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio
        )
        self.drop_path = nn.Dr
