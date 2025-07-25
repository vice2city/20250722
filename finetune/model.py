import torch
import torch.nn as nn
from torch.nn import LayerNorm, BatchNorm2d, Conv2d, Dropout, GELU

class MultiHeadAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1) @ v
        return self.out_proj(attn)

class DWConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dwconv = Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

    def forward(self, x):
        return self.dwconv(x)

class Mlp(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = Conv2d(dim, mlp_dim, 1)
        self.dwconv = DWConv(mlp_dim)
        self.act = GELU()
        self.fc2 = Conv2d(mlp_dim, dim, 1)
        self.drop = Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.drop(x)

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = BatchNorm2d(dim)
        self.attn = MultiHeadAttention(dim)
        self.norm2 = BatchNorm2d(dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LSKNet_multihead(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed1 = nn.Conv2d(3, 64, 7, 4, 3)
        self.block1 = nn.ModuleList([Block(64) for _ in range(2)])

    def forward(self, x):
        x = self.patch_embed1(x)
        for block in self.block1:
            x = block(x)
        return x

class OrientedRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = LSKNet_multihead()

    def forward(self, x):
        return self.backbone(x)
