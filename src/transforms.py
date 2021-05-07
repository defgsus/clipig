from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid


class RepeatTransform(torch.nn.Module):
    def __init__(self, count: Sequence[int]):
        super().__init__()
        self.count = count

    def forward(self, image):
        return image.repeat(1, 1, self.count[0]).repeat(1, self.count[1], 1)


class NoiseTransform(torch.nn.Module):
    def __init__(self, std: Sequence[float]):
        super().__init__()
        self.std = torch.nn.Parameter(
            torch.Tensor(std),
            requires_grad=False,
        )

    def forward(self, image):
        noise = torch.randn(image.shape).to(image.device)
        return image + noise * self.std.reshape(3, 1, 1)


class EdgeTransform(torch.nn.Module):
    def __init__(self, std: Sequence[float]):
        super().__init__()
        self.std = torch.nn.Parameter(
            torch.Tensor(std),
            requires_grad=False,
        )

    def forward(self, image):
        edge = VF.gaussian_blur(image, [5, 5], [2., 2.])
        edge = torch.clamp((image - edge) * 10., 0, 1)
        return edge

