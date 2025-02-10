import torch
from torch import nn

from MultiCNN import  MultiCNNBlock
# from modules.DDPM.Unet_Block import ResnetBlock, DiagonalGaussianDistribution
import math
from einops import rearrange
from torch.nn.init import trunc_normal_

from modules.DDPM.Unet_Block import ResnetBlock
from modules.autoencoderKL.distribution import DiagonalGaussianDistribution


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Latent_Encoder(nn.Module):
    def __init__(self, in_channels, layer_per_block, hidden_dims):
        super().__init__()
        self.in_channels = in_channels
        self.layer_per_block = layer_per_block
        self.hidden_dims = hidden_dims
        modules = []
        for h_dim in self.hidden_dims:
            for j in range(self.layer_per_block):
                modules.append(ResnetBlock(self.in_channels, dropout=0.0))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
            for j in range(self.layer_per_block):
                modules.append(ResnetBlock(in_channels, dropout=0.0))

        self.encoder = nn.Sequential(*modules)
        self.latent_encoder = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], 1)

    def forward(self, x):
        x = self.encoder(x)
        dist = DiagonalGaussianDistribution(self.latent_encoder(x))
        return dist


def exponential_linspace_int(dim_out, target_width, num, divisible_by):
    pass


class Sequence_Encoder(nn.Module):
    # 三维变成四维，增加空间信息维度
    def __init__(self, evo_dim=20):
        super().__init__()
        self.evo_dim = evo_dim,
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=self.evo_dim,
            kernel_size=1,
            stride=1)

    def forward(self, x, x_pad=None):
        # x:[B,C,L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, C, L] -> [B, C, 1, L]
        x = rearrange(x, 'b c r l -> (b c) r l')  # [B, C, 1, L] -> [B*C, 1, L]
        # 每个序列有一个12*20的张量表示，列代表进化信息，行代表与序列中其他位置的关联信息
        # 相当于增加一个空间维度信息，赋予接触图的概念
        x_emb = self.conv(
            x_pad)  # [B*C, 1, L] -> [B*C, 20, L] x_emb = rearrange(x_emb, '(b c) d n -> b c d n', b=B)  # [B*C, L, L] -> [B, C, L, L]
        # [64,21,20,12]
        return x_emb  # x_emb: [B, C, 20, L]


class MultiCNNEncoder(nn.Module):
    def __init__(self, filter_list, kernel_size=None):
        super().__init__()
        # 多尺度编码，只改变通道数，不改变尺寸 21->target_width
        # filter_list中通道数需要与kernel_size相匹配
        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                torch.nn.Sequential(
                    MultiCNNBlock(dim_in, dim_out, kernel_size),  #
                    Residual(MultiCNNBlock(dim_out, dim_out, [1])),
                ))
        self.conv_tower = torch.nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_tower:
            x = layer(x)
        return x


class Encoder():
    def __init__(self, target_width, num_layers, dim_out, layer_per_block, evo_dim, filter_list=[1, 3, 5]):
        # target_width是隐变量维度
        super(Encoder, self).__init__()
        self.dim_out = dim_out  # 结束多核卷积之后的维度
        self.target_width = target_width
        self.evo_dim = evo_dim
        self.tower_filter_list = exponential_linspace_int(
            self.dim_out, self.target_width, num=(num_layers + 1), divisible_by=2, )
        self.layer_per_block = layer_per_block
        self.filter_list = filter_list
        self.seq_layer = Sequence_Encoder(self.evo_dim)
        self.multiCNN_layer = MultiCNNEncoder(filter_list)
        self.latent_layer = Latent_Encoder(self.dim_out, self.layer_per_block, self.tower_filter_list)

    def forward(self, x):
        # 输入：[B,C,L]
        x = self.seq_layer(x)  # [B,C,L1,L2]-->[B,C,20,12]
        x = self.multiCNN_layer(x)  # [B,dim_out,20,12]或[B,20,12]
        x = self.latent_layer(x)  # diag
        return x
