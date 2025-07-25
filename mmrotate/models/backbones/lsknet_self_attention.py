import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
import warnings
from mmcv.cnn import build_norm_layer

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        # attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = (attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1))
        attn = self.conv(attn)
        return (x * attn)


# class Attention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.proj_1 = nn.Conv2d(d_model, d_model, 1)
#         self.activation = nn.GELU()
#         self.proj_2 = nn.Conv2d(d_model, d_model, 1)

#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         # 1. 计算查询、键和值
#         query = self.proj_1(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
#         key = self.proj_1(x).view(B, C, -1)  # (B, C, H*W)
#         value = x.view(B, C, -1)  # (B, C, H*W)

#         # 2. 计算注意力权重
#         attn_weights = torch.bmm(query, key)  # (B, H*W, H*W)
#         attn_weights = F.softmax(attn_weights, dim=-1)

#         # 3. 应用注意力权重
#         out = torch.bmm(attn_weights, value.permute(0, 2, 1))  # (B, H*W, C)
#         out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

#         # 4. 经过最终卷积
#         out = self.proj_2(out)
#         return out + x  # 残差连接

# class SelfAttention(nn.Module):
#     def __init__(self, dim_q, dim_k, dim_v):
#         super(SelfAttention, self).__init__()
#         self.dim_q = dim_q
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         # 定义线性变换函数
#         self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
#         self._norm_fact = 1 / math.sqrt(dim_k)
 
#     def forward(self, x):
#         # x: batch, n, dim_q
#         # 根据文本获得相应的维度
#         batch, n, dim_q = x.shape
#         # 如果条件为 True，则程序继续执行；如果条件为 False，则程序抛出一个 AssertionError 异常，并停止执行。
#         assert dim_q == self.dim_q  # 确保输入维度与初始化时的dim_q一致
        
#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v
        
#         # q*k的转置并除以开根号后的dim_k
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
#         # 归一化获得attention的相关系数
#         dist = F.softmax(dist, dim=-1)  # batch, n, n
#         # attention系数和v相乘，获得最终的得分
#         att = torch.bmm(dist, v)
#         return att

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 独立的投影层
        self.query_proj = nn.Conv2d(d_model, d_model, 1)
        self.key_proj = nn.Conv2d(d_model, d_model, 1)
        self.value_proj = nn.Conv2d(d_model, d_model, 1)
        
        self.activation = nn.GELU()
        self.proj_out = nn.Conv2d(d_model, d_model, 1)  # 输出层

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 计算查询、键和值
        query = self.query_proj(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        key = self.key_proj(x).view(B, C, -1)  # (B, C, H*W)
        value = self.value_proj(x).view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        # 2. 计算注意力权重并缩放
        d_k = query.size(-1)
        attn_weights = torch.bmm(query, key) / math.sqrt(d_k)  # (B, H*W, H*W)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 3. 应用注意力权重
        out = torch.bmm(attn_weights, value)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)  # (B, C, H, W)

        # 4. 经过输出投影层并添加残差连接
        out = self.proj_out(out)
        return out + x  # 残差连接



class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=None):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.attn = SelfAttention(dim)
        # self.attn = MultiHeadAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 引入 Layer Scale 参数
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = (x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x))))  # 使用 layer_scale
        x = (x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x))))  # 使用 layer_scale
        return x



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class LSKNet_self_attention(BaseModule):
    def __init__(self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], num_stages=4, 
                 pretrained=None,
                 init_cfg=None,
                 norm_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i], norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],norm_cfg=norm_cfg)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        # print("__init__")



    def init_weights(self):
        # print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(LSKNet_self_attention, self).init_weights()
        # print("init_weights")

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False
        # print("freeze_patch_emb")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        # print("get_classifier")
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        
        # print("forward_features!")
        # import ipdb;ipdb.set_trace()
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        # print("forward")
        # print(len(x))
        # print(x[0].size())
        # print(x[0])
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

