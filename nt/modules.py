"""Network modules that Ultralytics does not ship, for the model yamls in configs/models (M5 attention A/B).

Ultralytics resolves a yaml module name in the globals of ultralytics.nn.tasks, so call register() before
building a model from a yaml that uses one of these. Checkpoints pickle the class as nt.modules.<Name>, so loading
a trained .pt only needs nt to be importable.

Modules outside Ultralytics' own tables receive their yaml arguments unchanged and keep the input channel count,
so the channel argument in the yaml is the real, already width-scaled number for the model scale.
"""
from __future__ import annotations

import torch
from torch import nn


class CoordAtt(nn.Module):
    """Coordinate attention (Hou, Zhou and Feng, CVPR 2021): average-pool along each image axis, mix the two
    profiles with a shared 1x1 convolution, and gate the input per channel and per row and column.

    Built from mean, 1x1 convolution, batch norm, hardswish, sigmoid and multiply only, without the softmax and
    matrix products of C2PSA's self-attention. yaml: [-1, 1, CoordAtt, [channels, reduction]].
    """

    def __init__(self, c: int, reduction: int = 32):
        super().__init__()
        mip = max(8, c // reduction)
        self.conv1 = nn.Conv2d(c, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c, 1)
        self.conv_w = nn.Conv2d(mip, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        x_h = x.mean(3, keepdim=True)  # N, C, H, 1
        x_w = x.mean(2, keepdim=True).transpose(2, 3)  # N, C, W, 1
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], 2))))
        y_h, y_w = torch.split(y, [h, w], 2)
        return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w.transpose(2, 3)).sigmoid()


def register() -> None:
    """Make the modules above usable by name in Ultralytics model yamls."""
    from ultralytics.nn import tasks

    for cls in (CoordAtt,):
        setattr(tasks, cls.__name__, cls)
