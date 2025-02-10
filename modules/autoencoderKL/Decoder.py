from torch import nn
from MultiCNN import MultiCNNDecoder
from FinalBlock import FinalBlock
from modules.DDPM.Unet_Block import ResnetBlock


class MultiCNNDecoder(nn.Module):
	def __init__(self,kernel_size=[1,3,5],target_width=128,num_layers=3,):
		super().__init__()
		#多尺度编码，只改变通道数，不改变尺寸 21->target_width
		#filter_list中通道数需要与kernel_size相匹配
		self.target_width = target_width
		self.num_layers = num_layers
		self.kernel_size=kernel_size #多核卷积尺寸列表
		conv_layers = []
		tower_filter_list = exponential_linspace_int(
			self.target_width//2,
			self.target_width,
			num=self.num_layers+1,
			divisible_by=2,)
		tower_filter_list.reverse()
		for dim_in,dim_out in zip(tower_filter_list[:-1],tower_filter_list[1:]):
			conv_layers.append(
				torch.nn.Sequential(
					MultiCNNBlock(dim_in,dim_out,kernel_size),
					Residual(MultiCNNBlock(dim_out,dim_out,[1])),
				 ))
		self.conv_tower = torch.nn.ModuleList(conv_layers)
		self.feature_layer = MultiCNNBlock(self.target_width//2,21,[1])

	def forward(self, x):
		for layer in self.conv_tower:
			x = layer(x)
		#[B,21,L]
		x = self.feature_layer(x)
		return x

class Latent_Decoder(nn.Module):
    def __init__(self, hidden_dims, layer_per_block):
        super().__init__()
        hidden_dims.reverse()
        self.hidden_dims = hidden_dims
        self.layer_per_block = layer_per_block
        self.latent_decoder = nn.Conv2d(self.hidden_dims[-1] // 2, self.hidden_dims[-1], kernel_size=1)
        modules = []
        for i in range(len(hidden_dims) - 1):
            for j in range(self.layer_per_block):
                modules.append(ResnetBlock(hidden_dims[i], dropout=0.0))
            modules.append(
                nn.Sequential(
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i + 1],
                              kernel_size=3, stride=1, padding=1, ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()),
            )
            for j in range(self.layer_per_block):
                modules.append(ResnetBlock(hidden_dims[-1], dropout=0.0))
        self.decoder = nn.Sequential(*modules)

class FinalBlock(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        hidden_dims.reverse()  #########
        self.final_layer = nn.Sequential(
            nn.Conv2d(in_dims, in_dims, 3, 1, 1),
            nn.BatchNorm2d(in_dims),
            nn.LeakyReLU(),
            nn.Conv2d(in_dims, out_channels=1,
                      kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.final_layer(x)
        return x

class Decoder():
    def __init__(self, layer_per_block, kernel_size=[1, 3, 5], target_width=128, num_layers=3, hidden_dims=[]):
        super(Decoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.layer_per_block = layer_per_block  # the num of resblock
        self.kernel_size = kernel_size
        self.target_width = target_width  # 除2得到的是隐藏层z维度
        self.num_layers = num_layers
        self.latent_layer = Latent_Decoder(self.hidden_dims, self.layer_per_block)
        self.multiCNN_layer = MultiCNNDecoder(self.kernel_size, self.target_width, self.num_layers)
        self.final_layer = FinalBlock(self.target_width // 2)

    def forward(self, z):
        x = self.latent_layer(z)
        x = self.multiCNN_layer(x)
        x = self.final_layer(x)
        return x
