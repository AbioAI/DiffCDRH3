import math

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from modules.utils import default, exists


class MultiCNNBlock(nn.Module):
	def __init__(self, dim=21, dim_out=36, evo_dim=20,
				 kernel_sizes=[1,3,5],
				 gelu=True,
				 norm_type="group",
				 linear=None,
				 dilation=1):
		super().__init__()
		self.evo_dim = evo_dim,
		self.linear = linear,
		dim_out = default(dim_out,dim)
		original_dim_out = dim_out
		dim_out = dim_out//len(kernel_sizes)
		assert dim_out*len(kernel_sizes)==original_dim_out, "dim_out must be divisible by the number of kernel sizes."
		self.conv_layers = nn.ModuleList(
				[
					nn.Conv2d(
						dim,
						dim_out,
						kernel_size=kernel_size,
						padding=kernel_size // 2 if dilation == 1 else dilation,
						groups=1,
						dilation=dilation,
					)
					for kernel_size in self.kernel_sizes
				]
			)
		self.norm = None
		if norm_type == "group":
			self.norm = nn.GroupNorm(dim, dim)
		elif norm_type == "batch":
			self.norm = nn.BatchNorm2d(dim)
		self.gelu = nn.GELU() if gelu else None
		if self.linear is not None:
			self.linear_head = nn.Linear(original_dim_out*self.evo_dim*L, L*self.evo_dim)###########缺少L的定义
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, torch.nn.Linear):
			trunc_normal_(m.weight, std=0.02)
			if isinstance(m, torch.nn.Linear) and m.bias is not None:
				torch.nn.init.constant_(m.bias, 0)
		elif isinstance(m, torch.nn.LayerNorm):
			torch.nn.init.constant_(m.bias, 0)
			torch.nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, torch.nn.Conv2d):
			fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, torch.nn.Conv1d):
			fan_out = m.kernel_size[0] * m.out_channels
			fan_out //= m.groups
			m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.GroupNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
		elif isinstance(m, nn.Sequential):
			for submodule in m.children():
				self._init_weights(submodule)
		elif isinstance(m, nn.ModuleList):
			for submodule in m:
				self._init_weights(submodule)

	def forward(self, x):
		#x_emb: [B, C, 20, L]
		# fix: x.shape = (batch, seq_len, dim) if x.shape = (batch, 1, dim, seq_len)
		#x = rearrange(x, "b s d -> b d s")
		B,_,L = x.shape
		if exists(self.norm):
			x = self.norm(x)
		if exists(self.gelu):
			x = self.gelu(x)
		x = torch.cat([conv(x) for conv in self.conv_layers], dim=1)
		#x = rearrange(x, "b d s -> b s d")
		if exists(self.linear_head):
			# rearrange x from (batch, dim, seq_len) to (batch, seq_len, dim)
			x = x.view(x.size(0),-1)
			x = self.linear_head(x)
			x = x.view(x.size(0),self.evo_dim,L)
		return x #[64,20,12] 或 [64,dim_out,20,12]

