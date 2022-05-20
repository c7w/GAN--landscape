import jittor as jt
import jittor.nn as nn

from models.Transformer.SegFormer.DropPath import DropPath
from models.Transformer.SegFormer.SegFormerAttention import SegFormerAttention
from models.Transformer.SegFormer.SegFormerMLP import SegFormerMLP
from models.utils.init_weights import init_weights


class SegFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, proj_drop=0.1, attn_drop=0.1, drop_path=0.0, activation_func=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1.0):
        super(SegFormerBlock, self).__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SegFormerAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, sr_ratio=sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SegFormerMLP(
            in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
            active_function=activation_func, drop=proj_drop
        )

        self.apply(init_weights)

    def execute(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x
