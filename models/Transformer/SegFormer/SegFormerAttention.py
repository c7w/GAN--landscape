import math

import jittor as jt
import jittor.nn as nn


class SegFormerAttention(nn.Module):
    """
    Basic Attention Block.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1, sr_ratio=1):
        super(SegFormerAttention, self).__init__()
        assert dim % num_heads == 0, f"In SegFormer Attention: dim {dim} should be divided by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim / num_heads
        self.qk_scale = qk_scale if qk_scale is not None else math.pow(self.head_dim, -0.5)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # sequence reduction process
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self.init_weights)

    def execute(self, x, H, W):
        B, N, C = x.shape

        # B, self.num_heads, N, self.head_dim
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)  # B, N/(sr_ratio^2), C
            x_ = self.norm(x_)

            # B, self.num_heads, N/(sr_ratio^2), self.head_dim
            k = self.k(x_).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        else:
            k = self.k(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.qk_scale  # q * k^T, (B, self.num_heads, N, N/(sr_ratio^2))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, self.num_heads, N, self.head_dim) => (B, N, self.num_heads, self.head_dim) => (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
