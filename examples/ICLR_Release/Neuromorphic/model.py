import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven import layer, surrogate
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial

from spikeDE.neuron import LIFNeuron
from spikeDE import SNNWrapper
from spikeDE.layer import ClassificationHead

__all__ = ['spikformer']


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.,
                 tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.fc2_conv = nn.Conv1d(hidden_features, out_features, kernel_size=1, stride=1)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, N]
        x = self.fc1_conv(x)
        x = self.fc1_bn(x)
        x = self.fc1_lif(x)

        x = self.fc2_conv(x)
        x = self.fc2_bn(x)
        x = self.fc2_lif(x)
        return x


class SSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., sr_ratio=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25

        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

        self.attn_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, N]
        B, C, N = x.shape

        # Q path
        q = self.q_conv(x)
        q = self.q_bn(q)
        q = self.q_lif(q)
        q = q.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # K path
        k = self.k_conv(x)
        k = self.k_bn(k)
        k = self.k_lif(k)
        k = k.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # V path
        v = self.v_conv(x)
        v = self.v_bn(v)
        v = self.v_lif(v)
        v = v.transpose(-1, -2).reshape(B, N, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3).contiguous()  # [B, num_heads, N, C//num_heads]

        # Attention
        attn = (q @ k.transpose(-2, -1)) @ v  # [B, num_heads, N, C//num_heads]
        x = attn * self.scale

        # Reshape back
        x = x.transpose(2, 3).reshape(B, C, N).contiguous()
        x = self.attn_lif(x)

        # Output projection
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 sr_ratio=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
                        tau=tau, threshold=threshold, surrogate_grad_scale=surrogate_grad_scale)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                       tau=tau, threshold=threshold, surrogate_grad_scale=surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, N]
        identity = x
        x = self.attn(x)
        x = x + identity  # First residual

        identity = x
        x = self.mlp(x)
        # Don't add residual here, let MLP output directly
        # x = x + identity  # Remove this line
        return x + identity  # Keep the residual
    #
    # def forward(self, x):
    #     # Input: [B, C, N]
    #     x = x + self.attn(x)
    #     x = x + self.mlp(x)
    #     return x


class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2,
                 embed_dims=256, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()

        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims // 4)
        self.proj_lif1 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims // 4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims // 2)
        self.proj_lif2 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = LIFNeuron(tau, threshold, surrogate_grad_scale)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = LIFNeuron(tau, threshold, surrogate_grad_scale)

    def forward(self, x):
        # Input: [B, C, H, W]
        B, C, H, W = x.shape

        # Stage 1
        x = self.proj_conv(x)
        x = self.proj_bn(x)
        x = self.proj_lif(x)
        x = self.maxpool(x)

        # Stage 2
        x = self.proj_conv1(x)
        x = self.proj_bn1(x)
        x = self.proj_lif1(x)
        x = self.maxpool1(x)

        # Stage 3
        x = self.proj_conv2(x)
        x = self.proj_bn2(x)
        x = self.proj_lif2(x)
        x = self.maxpool2(x)

        # Stage 4
        x = self.proj_conv3(x)
        x = self.proj_bn3(x)
        x = self.proj_lif3(x)
        x = self.maxpool3(x)

        # Relative position encoding
        x_rpe = self.rpe_conv(x)
        x_rpe = self.rpe_bn(x_rpe)
        x_rpe = self.rpe_lif(x_rpe)
        x = x + x_rpe

        # Reshape to [B, embed_dims, num_patches]
        B, C, H_out, W_out = x.shape
        x = x.reshape(B, C, H_out * W_out).contiguous()

        return x





class Spikformer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=256, num_heads=16, mlp_ratios=4, qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=2, sr_ratios=1, tau=2.0, threshold=1.0, surrogate_grad_scale=5.0):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        self.patch_embed = SPS(img_size_h=img_size_h, img_size_w=img_size_w,
                               patch_size=patch_size, in_channels=in_channels,
                               embed_dims=embed_dims, tau=tau, threshold=threshold,
                               surrogate_grad_scale=surrogate_grad_scale)

        self.block = nn.ModuleList([
            Block(dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                  attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                  sr_ratio=sr_ratios, tau=tau, threshold=threshold,
                  surrogate_grad_scale=surrogate_grad_scale)
            for j in range(depths)
        ])

        self.head = ClassificationHead(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Input: [B, C, H, W]
        x = self.patch_embed(x)  # [B, C, N]

        for blk in self.block:
            x = blk(x)

        # Global average pooling and classification
        x = self.head(x)
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    # 提取 SNNWrapper 需要的参数
    integrator = kwargs.pop('integrator', 'fdeint')
    alpha = kwargs.pop('alpha', None)
    beta = kwargs.pop('beta', 0.5)
    multi_coefficient = kwargs.pop('multi_coefficient', None)
    learn_coefficient = kwargs.pop('learn_coefficient', False)
    device = kwargs.pop('device', 'cuda:0')
    num_classes = kwargs.pop('num_classes',101)
    # 过滤掉其他不被 Spikformer 接受的参数
    kwargs.pop('method', None)
    kwargs.pop('step_size', None) 
    kwargs.pop('time_interval', None)
    kwargs.pop('memory', None)
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    
    model = Spikformer(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        tau=2.0, threshold=1.0, surrogate_grad_scale=5.0,
        **kwargs
    )
    model.default_cfg = _cfg()
    
    # 包装成 SNNWrapper
    if alpha is None:
        alpha = beta
    net = SNNWrapper(
        model,
        integrator=integrator,
        alpha=alpha,
        multi_coefficient=multi_coefficient,
        learn_coefficient=learn_coefficient,
    ).to(device)

    # 设置神经元形状，输入格式为 [N, C, H, W]
    net._set_neuron_shapes(input_shape=(1, 2, 128, 128))
    return net
