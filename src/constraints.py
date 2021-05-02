from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .expression import Expression, ExpressionContext


class ConstraintBase(torch.nn.Module):

    def __init__(
            self,
            weight: Union[float, Expression],
    ):
        super().__init__()
        self.weight = weight


class AboveBelowConstraintBase(ConstraintBase):

    def __init__(
            self,
            weight: Union[float, Expression],
    ):
        super().__init__(weight=weight)

    def get_image_value(self, image: torch.Tensor):
        raise NotImplementedError

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        value = self.get_image_value(image)

        loss_sum = torch.tensor(0)

        if self.above is not None:
            target = torch.tensor(context(self.above)).to(image.device)
            loss_sum = loss_sum + torch.clamp_min(target - value, 0).pow(2).mean()

        if self.below is not None:
            target = torch.tensor(context(self.below)).to(image.device)
            loss_sum = loss_sum + torch.clamp_min(value - target, 0).pow(2).mean()

        return context(self.weight) * loss_sum

    def TODO_extra_repr(self) -> str:
        t = ""
        if self.target_above is not None:
            t = "above=[%s]" % ",".join(str(round(float(f), 2)) for f in self.target_above)
        if self.target_below is not None:
            if t:
                t += ", "
            t = "below=[%s]" % ",".join(str(round(float(f), 2)) for f in self.target_below)
        return t


class AboveBelow3ConstraintBase(AboveBelowConstraintBase):

    def __init__(
            self,
            weight: Union[float, Expression],
            above: List[Union[float, Expression]] = None,
            below: List[Union[float, Expression]] = None
    ):
        super().__init__(weight=weight)
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"
        self.below = below
        self.above = above


class AboveBelow1ConstraintBase(AboveBelowConstraintBase):

    def __init__(
            self,
            weight: Union[float, Expression],
            above: Union[float, Expression] = None,
            below: Union[float, Expression] = None
    ):
        super().__init__(weight=weight)
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"
        self.below = below
        self.above = above


class MeanConstraint(AboveBelow3ConstraintBase):

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.mean()


class StdConstraint(AboveBelow3ConstraintBase):

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.std()


class SaturationConstraint(AboveBelow1ConstraintBase):

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return get_mean_saturation(image)


def get_mean_saturation(image: torch.Tensor) -> torch.Tensor:
    color_planes = image.reshape(3, -1)
    mean_plane = color_planes.mean(dim=0, keepdim=True)
    saturation_plane = torch.abs(mean_plane.repeat(3, 1) - color_planes).sum(0, keepdim=True) / 3.
    return saturation_plane.mean()
