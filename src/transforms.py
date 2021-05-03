from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid


class RepeatTransform(torch.nn.Module):
    def __init__(self, resolution: Sequence[int]):
        super().__init__()
        self.resolution = resolution

    def forward(self, image):
        return make_grid(
            image.unsqueeze(0).repeat(self.resolution[0] * self.resolution[1], 1, 1, 1),
            nrow=self.resolution[0],
            padding=0,
        )


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

