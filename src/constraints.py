from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .expression import Expression, ExpressionContext
from .parameters import Parameter, SequenceParameter, EXPR_ARGS


constraints = dict()


class ConstraintBase(torch.nn.Module):
    NAME = None
    PARAMS = None

    def __init_subclass__(cls, **kwargs):
        if cls.NAME is not None:
            constraints[cls.NAME] = cls

    def __init__(
            self,
            weight: Union[float, Expression],
    ):
        super().__init__()
        self.weight = weight

    def description(self, context: ExpressionContext) -> str:
        raise NotImplementedError


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

        return context(self.weight) * 100. * loss_sum


class AboveBelow3ConstraintBase(AboveBelowConstraintBase):
    PARAMS = {
        "weight": Parameter(float, default=1., expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "above": SequenceParameter(float, length=3, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "below": SequenceParameter(float, length=3, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    }

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

    def description(self, context: ExpressionContext) -> str:
        text = ""
        if self.above:
            value = context(self.above)
            text = f"above=[%s]" % ", ".join(str(round(float(f), 2)) for f in value)
        if self.below:
            if text:
                text += ", "
            value = context(self.below)
            text += f"below=[%s]" % ", ".join(str(round(float(f), 2)) for f in value)
        return text


class AboveBelow1ConstraintBase(AboveBelowConstraintBase):
    PARAMS = {
        "weight": Parameter(float, default=1., expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "above": Parameter(float, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "below": Parameter(float, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    }

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

    def description(self, context: ExpressionContext) -> str:
        text = ""
        if self.above:
            value = context(self.above)
            text = f"above={value:.3f}"
        if self.below:
            if text:
                text += ", "
            value = context(self.below)
            text += f"below={value:.3f}"
        return text


class MeanConstraint(AboveBelow3ConstraintBase):
    NAME = "mean"

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.mean()

    def description(self, context: ExpressionContext) -> str:
        return f"mean({super().description(context)})"


class StdConstraint(AboveBelow3ConstraintBase):
    NAME = "std"

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.std()

    def description(self, context: ExpressionContext) -> str:
        return f"std({super().description(context)})"


class SaturationConstraint(AboveBelow1ConstraintBase):
    NAME = "saturation"

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return get_mean_saturation(image)

    def description(self, context: ExpressionContext) -> str:
        return f"sat({super().description(context)})"


def get_mean_saturation(image: torch.Tensor) -> torch.Tensor:
    color_planes = image.reshape(3, -1)
    mean_plane = color_planes.mean(dim=0, keepdim=True)
    saturation_plane = torch.abs(mean_plane.repeat(3, 1) - color_planes).sum(0, keepdim=True) / 3.
    return saturation_plane.mean()


class BlurConstraint(ConstraintBase):
    NAME = "blur"
    PARAMS = {
        "weight": Parameter(float, default=1., expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "kernel_size": SequenceParameter(int, length=2, default=[3, 3], expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "sigma": SequenceParameter(float, length=2, null=True, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    }

    def __init__(
            self,
            weight: Union[float, Expression],
            kernel_size: List[Union[int, Expression]] = (3, 3),
            sigma: List[Union[float, Expression]] = (.5, .5),
    ):
        super().__init__(weight=weight)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        kernel_size = [int(k) for k in context(self.kernel_size)]
        sigma = context(self.sigma)

        blurred_image = VF.gaussian_blur(image, kernel_size, sigma)

        loss = F.mse_loss(
            image.reshape(3, -1),
            blurred_image.reshape(3, -1),
        )

        return context(self.weight) * 100. * loss

    def description(self, context: ExpressionContext) -> str:
        kernel_size = [int(k) for k in context(self.kernel_size)]
        sigma = [round(f, 3) for f in context(self.sigma)]
        return f"blur(ks={kernel_size}, sigma={sigma})"


class EdgeMeanConstraint(AboveBelow3ConstraintBase):
    NAME = "edge_mean"

    def get_image_value(self, image: torch.Tensor):
        return get_edge_mean(image)

    def description(self, context: ExpressionContext) -> str:
        return f"edge_mean({super().description(context)})"


class EdgeMaxConstraint(AboveBelow3ConstraintBase):
    NAME = "edge_max"

    def get_image_value(self, image: torch.Tensor):
        return get_edge_max(image)

    def description(self, context: ExpressionContext) -> str:
        return f"edge_max({super().description(context)})"


def get_edge_mean(image: torch.Tensor) -> torch.Tensor:
    blurred_image = VF.gaussian_blur(image, [5, 5], None)
    edges = (blurred_image - image)
    edges = torch.abs(edges)
    return edges.reshape(3, -1).mean(1)


def get_edge_max(image: torch.Tensor) -> torch.Tensor:
    blurred_image = VF.gaussian_blur(image, [5, 5], None)
    edges = (blurred_image - image)
    edges = torch.abs(edges.reshape(3, -1))
    edge_max = torch.max(edges, 1).values * .7
    edges_mean = torch.cat([
        edges[0, edges[0] > edge_max[0]].mean().unsqueeze(0),
        edges[1, edges[1] > edge_max[1]].mean().unsqueeze(0),
        edges[2, edges[2] > edge_max[2]].mean().unsqueeze(0),
    ])
    return edges_mean
