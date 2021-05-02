from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .expression import Expression, ExpressionContext


class AboveBelowConstraintBase(torch.nn.Module):

    def __init__(
            self,
            weight: Union[float, Expression],
            above: List[Union[float, Expression]] = None,
            below: List[Union[float, Expression]] = None
    ):
        super().__init__()
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"

        self.weight = weight
        self.below = below
        self.above = above

    def TODO_extra_repr(self) -> str:
        t = ""
        if self.target_above is not None:
            t = "above=[%s]" % ",".join(str(round(float(f), 2)) for f in self.target_above)
        if self.target_below is not None:
            if t:
                t += ", "
            t = "below=[%s]" % ",".join(str(round(float(f), 2)) for f in self.target_below)
        return t

    def get_image_value(self, image: torch.Tensor):
        raise NotImplementedError

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        value = self.get_image_value(image)

        loss_sum = torch.tensor(0)

        for target in (self.above, self.below):
            if target is not None:
                target = context(target)
                target = torch.Tensor(target).to(image.device)
                loss_sum = loss_sum + torch.clamp_min(target - value, 0).pow(2).mean()

        return context(self.weight) * loss_sum


class MeanConstraint(AboveBelowConstraintBase):

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.mean()


class StdConstraint(AboveBelowConstraintBase):

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.std()
