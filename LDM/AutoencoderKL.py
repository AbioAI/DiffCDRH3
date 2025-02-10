import torch

import pytorch_lightning as pl

from config import instantiate_from_config
from modules.autoencoderKL.Decoder import Decoder
from modules.autoencoderKL.Encoder import Encoder


class AutoencoderKL(pl.LightningModule):
    def __init__(self, config,  ckpt_path=None, ignore_keys=None):
        super().__init__()
        if ignore_keys is None:
            ignore_keys = []
        #target_width, num_layers, dim_out, layer_per_block,evo_dim, filter_list=[1, 3, 5]
        self.encoder = Encoder(**config)
        #layer_per_block, kernel_size = [1, 3, 5], target_width = 128, num_layers = 3, hidden_dims = []):

        self.decoder = Decoder(**config)
        self.loss = instantiate_from_config(**config)  ####

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def encode(self, x):
        dist = self.encoder(x)
        z = dist.sample()
        return dist, z

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, s):
        posterior = self.encode(input)
        dec = self.decode(posterior)
        return dec, posterior

    def get_input(self, batch):
        x = batch  #
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)

        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def get_last_layer(self):
        return self.decoder.conv_out.weight
