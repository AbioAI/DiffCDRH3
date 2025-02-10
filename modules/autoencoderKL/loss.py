import torch
import torch.nn as nn
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.lpips import LPIPS
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss
import torch
import torch.nn.functional as F


def calculate_entropy(prob_dist):
    """
    计算给定概率分布的熵值。

    :param prob_dist: 概率分布张量，形状为 [B, N]，其中 B 是批次大小，N 是类别数。
    :return: 熵值，形状为 [B]，即每个样本的熵值。
    """
    # 确保概率分布是合法的（值在0到1之间，并且归一化总和为1）
    prob_dist = F.softmax(prob_dist, dim=-1)
    log_prob = torch.log(prob_dist + 1e-10)
    entropy = -torch.sum(prob_dist * log_prob, dim=-1)

    return entropy


def EvoAdaptLoss(inputs, reconstructions, en, r_plus=1.0, r_minus=1.0, alpha=0.5):
    """
    自定义条件损失函数，根据 en 的值选择不同的损失计算方式。

    :param inputs: 目标张量，形状 [B, 12, 21]，one-hot 编码标签
    :param reconstructions: 重建输出，形状 [B, 12, 21]，logits
    :param en: 决定损失计算方式的条件张量或标量
    :param r_plus: 当 en >= 2.5 时的权重
    :param r_minus: 当 en < 2.5 时的权重
    :param alpha: 用于调整损失计算的比例参数
    :return: 计算后的损失值
    """

    # 将 inputs 转换为类别索引
    inputs_flat = torch.argmax(inputs, dim=-1)  # [B, 12] -> 类别索引

    # 将 reconstructions 展平为 [B * 12, 21]，即每个位置的 21 个类别 logits
    reconstructions_flat = reconstructions.view(-1, 21)  # [B * 12, 21]

    # 根据 en 的值选择不同的损失计算方式
    # 如果 en >= 2.5 使用 r+ 和 P_t 的计算
    # 如果 en < 2.5 使用 r- 和 P_t 的计算

    # 计算 log(P_t)（logits）
    log_p_t = F.log_softmax(reconstructions_flat, dim=-1)  # 对 logits 进行 softmax 和 log 计算

    # 计算损失
    loss_pos = torch.sum(r_plus * inputs_flat * log_p_t, dim=-1)  # 如果 en >= 2.5
    loss_neg = torch.sum(alpha * (1 - inputs_flat) * log_p_t, dim=-1)  # 如果 en < 2.5

    # 根据 en 的条件选择损失
    loss = torch.where(en >= 2.5, loss_pos, loss_neg)  # 如果 en >= 2.5 用 loss_pos，否则用 loss_neg

    # 平均损失
    return loss.mean()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class TotalLoss(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, recloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=21, disc_factor=1.0, disc_weight=1.0,
                 use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):
        # disc_start:控制什么时间使用判别器
        # disc_factor:控制生成器训练和判别器训练之间平衡的因子
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.recloss_weight = recloss_weight
        # 初始化对数方差参数，用于控制重建损失的缩放
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,  # 未导入
                                                 n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss  # 未定义
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.rec_loss = EvoAdaptLoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        # 保留nll_loss和g_loss相对于最后一层梯度的范数，以此来衡量重要性
        nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)  # 衡量在梯度上的重要性
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # inputs即target:[B,12,21] reconstruction:[B,12,21]
        inputs_flat = torch.argmax(inputs, dims=-1)  # [B,12] 每个位置的类别索引
        inputs_flat = inputs_flat.view(-1)
        reconstructions_flat = reconstructions.view(-1, 21)  # [B*12,21]
        rec_loss = self.rec_loss(reconstructions_flat, inputs_flat)

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
