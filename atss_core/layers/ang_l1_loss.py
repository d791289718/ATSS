import torch
from torch import nn


class AngL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss( reduce=False, reduction="none")

    def forward(self, pred, target, weight=None):
        losses = self.loss(pred, target)
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()
