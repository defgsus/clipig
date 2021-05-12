from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .expression import Expression, ExpressionContext
from .parameters import Parameter, SequenceParameter, EXPR_ARGS


Int = Union[int, Expression]
Float = Union[float, Expression]


constraints = dict()


class ConstraintBase(torch.nn.Module):
    """
    So, constraint modules must implement a forward function
    which takes the image tensor and an ExpressionContext
    and which must return a scalar tensor with the loss.

    The constructor must accept all defined parameters
    except the `weight` parameter, which is handled
    by ImageTrainer.
    """
    NAME = None
    PARAMS = None

    def __init_subclass__(cls, **kwargs):
        if cls.NAME is not None:
            constraints[cls.NAME] = cls
        if cls.PARAMS is not None:
            cls.PARAMS = {
                "weight": Parameter(float, default=1.),
                **cls.PARAMS,
            }

    def forward(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        raise NotImplementedError

    def description(self, context: ExpressionContext) -> str:
        if not self.PARAMS:
            return f"{self.NAME}()"

        param_strs = []
        for name, param in self.PARAMS.items():
            if name == "weight":
                continue
            value = getattr(self, name)
            if value is None:
                continue
            value = context(value)
            if isinstance(value, float):
                value = str(round(value, 3))
            elif isinstance(value, list):
                if all(v == value[0] for v in value):
                    value = str(round(value[0], 3))
                else:
                    value = ", ".join(
                        str(round(v, 3) if isinstance(v, float) else v)
                        for v in value
                    )
                    value = f"[{value}]"
            else:
                value = str(value)
            param_strs.append(f"{name}={value}")

        param_strs = ", ".join(param_strs)
        return f"{self.NAME}({param_strs})"


class AboveBelowConstraintBase(ConstraintBase):

    WEIGHT_FACTOR = 1.

    def get_image_value(self, image: torch.Tensor):
        raise NotImplementedError

    def forward(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        value = self.get_image_value(image)

        loss_sum = torch.tensor(0)

        if self.above is not None:
            target = torch.tensor(context(self.above)).to(image.device)
            loss_sum = loss_sum + torch.clamp_min(target - value, 0).pow(2).mean()

        if self.below is not None:
            target = torch.tensor(context(self.below)).to(image.device)
            loss_sum = loss_sum + torch.clamp_min(value - target, 0).pow(2).mean()

        return loss_sum * self.WEIGHT_FACTOR


class AboveBelow3ConstraintBase(AboveBelowConstraintBase):
    PARAMS = {
        "above": SequenceParameter(float, length=3, default=None),
        "below": SequenceParameter(float, length=3, default=None),
    }

    def __init__(
            self,
            above: List[Union[float, Expression]] = None,
            below: List[Union[float, Expression]] = None
    ):
        super().__init__()
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"
        self.below = below
        self.above = above


class AboveBelow1ConstraintBase(AboveBelowConstraintBase):
    PARAMS = {
        "above": Parameter(float, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "below": Parameter(float, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    }

    def __init__(
            self,
            above: Union[float, Expression] = None,
            below: Union[float, Expression] = None
    ):
        super().__init__()
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"
        self.below = below
        self.above = above


class MeanConstraint(AboveBelow3ConstraintBase):
    NAME = "mean"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.mean()


class StdConstraint(AboveBelow3ConstraintBase):
    NAME = "std"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return image.std()


class SaturationConstraint(AboveBelow1ConstraintBase):
    NAME = "saturation"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        image = image.reshape(3, -1)
        return get_mean_saturation(image)


def get_mean_saturation(image: torch.Tensor) -> torch.Tensor:
    color_planes = image.reshape(3, -1)
    mean_plane = color_planes.mean(dim=0, keepdim=True)
    saturation_plane = torch.abs(mean_plane.repeat(3, 1) - color_planes).sum(0, keepdim=True) / 3.
    return saturation_plane.mean()


class BlurConstraint(ConstraintBase):
    NAME = "blur"
    PARAMS = {
        "kernel_size": SequenceParameter(int, length=2, default=[3, 3], expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
        "sigma": SequenceParameter(float, length=2, null=True, default=None, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    }

    def __init__(
            self,
            kernel_size: List[Union[int, Expression]] = (3, 3),
            sigma: List[Union[float, Expression]] = (.5, .5),
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        kernel_size = context(self.kernel_size)
        kernel_size = [
            max(1, k+1 if k % 2 == 0 else k)
            for k in kernel_size
        ]
        if self.sigma is None:
            sigma = None
        else:
            sigma = [max(0.0001, s) for s in context(self.sigma)]

        blurred_image = VF.gaussian_blur(image, kernel_size, sigma)

        loss = F.mse_loss(
            image.reshape(3, -1),
            blurred_image.reshape(3, -1),
        )

        return 100. * loss


class EdgeMeanConstraint(AboveBelow3ConstraintBase):
    NAME = "edge_mean"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        return get_edge_mean(image)

    def description(self, context: ExpressionContext) -> str:
        return f"edge_mean({super().description(context)})"


class EdgeMaxConstraint(AboveBelow3ConstraintBase):
    NAME = "edge_max"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        return get_edge_max(image)


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


class BorderConstraint(ConstraintBase):
    NAME = "border"
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=[1, 1]),
        "color": SequenceParameter(float, length=3, default=[1., 1., 1.]),
    }

    def __init__(self, size: List[Int], color: List[Float]):
        super().__init__()
        self.size = size
        self.color = color
        self.loss_function = torch.nn.L1Loss()

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        size = context(self.size)
        color = torch.Tensor(context(self.color)).to(image.device)
        image_border = image.clone()

        s = min(size[1], image.shape[1])
        color_s = color.reshape(3, 1, 1).repeat(1, s, 1)
        image_border[:, :s, :] = color_s
        image_border[:, -s:, :] = color_s

        s = min(size[0], image.shape[2])
        color_s = color.reshape(3, 1, 1).repeat(1, 1, s)
        image_border[:, :, :s] = color_s
        image_border[:, :, -s:] = color_s

        return 100. * self.loss_function(
            image, image_border
        )


class NormalizeConstraint(ConstraintBase):
    NAME = "normalize"
    PARAMS = {
        "min": SequenceParameter(float, length=3, default=[0., 0., 0.]),
        "max": SequenceParameter(float, length=3, default=[1., 1., 1.]),
    }

    def __init__(self, min: List[Float], max: List[Float]):
        super().__init__()
        self.min = min
        self.max = max
        self.loss_function = torch.nn.L1Loss()

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        mi = image.min(-1).values.min(-1).values.reshape(3, 1, 1)
        ma = image.max(-1).values.max(-1).values.reshape(3, 1, 1)
        diff = ma - mi
        diff[diff == 0] = 1.

        desired_min = torch.Tensor(context(self.min)).to(image.device).reshape(3, 1, 1)
        desired_max = torch.Tensor(context(self.max)).to(image.device).reshape(3, 1, 1)

        normed_image = (image - mi) / diff
        normed_image = desired_min + normed_image * (desired_max - desired_min)

        return 10. * self.loss_function(
            image, normed_image
        )


class ContrastConstraint(AboveBelow3ConstraintBase):
    NAME = "contrast"
    # WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor):
        mean = image.mean(-1).mean(-1).reshape(3, 1, 1)
        low_mean = image[image < mean].mean(-1).mean(-1)
        high_mean = image[image > mean].mean(-1).mean(-1)

        spread = high_mean - low_mean
        return spread
