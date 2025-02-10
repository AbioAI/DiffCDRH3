import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def cross_entropy_loss(y_pred, y_true):
    """
    计算交叉熵损失
    :param y_pred: 预测的概率分布（softmax输出），形状为 [batch_size, num_classes]
    :param y_true: 真实标签，形状为 [batch_size]
    :return: 计算得到的交叉熵损失
    """
    # 防止log(0)的情况
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

    # 计算交叉熵损失
    batch_size = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(batch_size), y_true])
    loss = np.sum(log_likelihood) / batch_size
    return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss: 用于处理类别不平衡问题。
        :param alpha: 平衡因子，控制各类别的权重，默认为0.25。
        :param gamma: 焦点因子，控制难易样本的关注度，默认为2。
        :param reduction: 'mean' 或 'sum'，决定返回的损失形式。'mean' 返回平均损失，'sum' 返回总损失。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        计算焦点损失
        :param inputs: 模型预测的 logits，形状为 (batch_size, num_classes)
        :param targets: 真实标签，形状为 (batch_size)
        :return: 焦点损失
        """

        probs = F.softmax(inputs, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1))
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        fl_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return fl_loss.mean()
        elif self.reduction == 'sum':
            return fl_loss.sum()
        else:
            return fl_loss


def l1(x, y):
    return torch.abs(x - y)


def l2(x, y):
    return torch.pow((x - y), 2)
