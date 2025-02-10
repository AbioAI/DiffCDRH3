from torch import nn
import torch
from torch._C._nn import linear
from Unet_Block import TimestepEmbedSequential, conv_nd, ResBlock, timestep_embedding, zero_module, SpatialTransformer, \
    normalization, AttentionBlock


class UNetModel(nn.Module):  # [B,target_width//2,20,12]

    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, dropout, channel_mult, conv_resample, dims, use_checkpoint, num_heads,
                 use_scale_shift_norm, num_heads_upsample,
                 content_dim=None
                 , context_dim=None, use_new_attention_order=None, transformer_depth=None, use_spatial_transformer=None,
                 n_embed=None):
        context_dim = list(context_dim)
        time_embed_dim = model_channels * 4

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.content_dim = content_dim

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ])

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels  # 模型的起始通道数
        # ds = 1 #记录下采样的倍数，该模型不进行下采样
        # channel_mult:通道扩展系数[1,2,4]表示三层，分别是1x,2x,4x的通道扩展
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,  # 网络维度，1D/2D/3D
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )]
                ch = mult * model_channels

                dim_head = ch // num_heads
                layers.append(
                    AttentionBlock(       ##############缺少attention块
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads,
                        num_head_channels=dim_head,
                        use_new_attention_order=use_new_attention_order,
                    ) if not use_spatial_transformer else SpatialTransformer(
                        ch, num_heads, dim_head, depth=transformer_depth, context_dim=self.context_dim
                    )
                )
                # ResBlock和AttentionBlock合并为input_blocks
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch  # 总特征数
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=False,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                layers.append(
                    AttentionBlock(
                        ch,
                        use_checkpoint=use_checkpoint,
                        num_heads=num_heads_upsample,
                        num_head_channels=dim_head,
                        use_new_attention_order=use_new_attention_order,
                    ) if not use_spatial_transformer else SpatialTransformer(
                        ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                    )
                )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=False,
                        )
                    )
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
                # nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
            )

    def forward(self, x, timesteps=None, context=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = h.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)
