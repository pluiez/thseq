import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import thseq.utils as utils


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """fairseq's approximation of label-smoothed cross entropy"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def label_smoothed_nll_loss_2(log_probs, target, eps=0.1, ignore_index=-100, reduce=True):
    """An accurate formulation of label-smoothed cross entropy"""
    if eps <= 0:
        return F.nll_loss(log_probs, target.flatten(), ignore_index=ignore_index, reduction='sum' if reduce else 'none')
    confidence = 1.0 - eps
    vocab_size = log_probs.size(-1)
    noise_confidence = eps / (vocab_size - 1)

    target = target.unsqueeze(-1)

    nll_loss = -log_probs.gather(dim=-1, index=target)
    smoothed_loss = -log_probs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        mask = target.ne(ignore_index)
        nll_loss = nll_loss[mask]
        smoothed_loss = smoothed_loss[mask]

    loss = smoothed_loss * noise_confidence + (confidence - noise_confidence) * nll_loss
    loss = loss.squeeze(-1)
    normalizing = -(confidence * math.log(confidence) + float(vocab_size - 1) *
                    noise_confidence * math.log(noise_confidence + 1e-20))
    loss -= normalizing
    if reduce:
        loss = loss.sum()
    return loss, nll_loss.sum()


class CrossEntropyLoss(nn.Module):
    def __init__(self, eps=0.1, weight=None, ignore_index=-100, reduce=True) -> None:
        super().__init__()
        reduce = reduce or 'none'
        self.eps = eps
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = utils.log_softmax(logit, -1)
        return label_smoothed_nll_loss(log_probs, target, self.eps, self.ignore_index, reduce=self.reduce)
