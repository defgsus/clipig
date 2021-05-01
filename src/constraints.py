from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid


class StdConstraint(torch.nn.Module):
    def __init__(self, weight: float, above: Sequence[float] = None, below: Sequence[float] = None):
        super().__init__()
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"

        self.weight = weight

        if above:
            self.target_above = torch.nn.Parameter(torch.Tensor(above), requires_grad=False)
        else:
            self.target_above = None

        if below:
            self.target_below = torch.nn.Parameter(torch.Tensor(below), requires_grad=False)
        else:
            self.target_below = None

    def forward(self, image: torch.Tensor):
        image = image.reshape(3, -1)

        std = image.std(1)

        loss_sum = torch.tensor(0)

        if self.target_above is not None:
            loss_sum = loss_sum + torch.clamp_min(self.target_above - std, 0).pow(2).mean()

        if self.target_below is not None:
            loss_sum = loss_sum + torch.clamp_min(std - self.target_below, 0).pow(2).mean()

        return self.weight * loss_sum
