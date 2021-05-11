import random
from typing import Union, Sequence, List, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid

from .expression import Expression, ExpressionContext
from .parameters import (
    Parameter, SequenceParameter, EXPR_ARGS
)

Int = Union[int, Expression]
Float = Union[float, Expression]


transformations = dict()


class TransformBase:
    NAME = None
    IS_RANDOM = False
    IS_RESIZE = False
    PARAMS = None

    def __init_subclass__(cls, **kwargs):
        assert cls.NAME, f"Must specify {cls.__name__}.NAME"
        assert cls.PARAMS, f"Must specify {cls.__name__}.PARAMS"
        transformations[cls.NAME] = cls

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        raise NotImplementedError


class Blur(TransformBase):
    NAME = "blur"
    PARAMS = {
        "kernel_size": SequenceParameter(int, length=2, default=[3, 3]),
        "sigma": SequenceParameter(float, length=2, null=True, default=None),
    }

    def __init__(self, kernel_size: List[Int], sigma: List[float]):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        kernel_size = context(self.kernel_size)
        kernel_size = [
            max(1, k+1 if k % 2 == 0 else k)
            for k in kernel_size
        ]
        if self.sigma is None:
            sigma = None
        else:
            sigma = [max(0.0001, s) for s in context(self.sigma)]
        return VF.gaussian_blur(image, kernel_size, sigma)


class Resize(TransformBase):
    NAME = "resize"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=None),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VF.resize(image, size)


class CenterCrop(TransformBase):
    NAME = "center_crop"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=None),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VF.center_crop(image, size, fill=None)


class RandomCrop(TransformBase):
    NAME = "random_crop"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=None),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VT.RandomCrop(size=size)(image)


class Repeat(TransformBase):
    NAME = "repeat"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=None),
    }

    def __init__(self, count: List[Int]):
        super().__init__()
        self.count = count

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        count = context(self.count)
        return image.repeat(1, 1, count[0]).repeat(1, count[1], 1)


class Border(TransformBase):
    NAME = "border"
    PARAMS = {
        "size": SequenceParameter(int, length=2, default=[1, 1]),
        "color": SequenceParameter(float, length=3, default=[1., 1., 1.]),
    }

    def __init__(self, size: List[Int], color: List[Float]):
        super().__init__()
        self.size = size
        self.color = color

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        color = torch.Tensor(context(self.color)).to(image.device)
        image = image.clone()

        s = min(size[1], image.shape[1])
        color_s = color.reshape(3, 1, 1).repeat(1, s, 1)
        image[:, :s, :] = color_s
        image[:, -s:, :] = color_s

        s = min(size[0], image.shape[2])
        color_s = color.reshape(3, 1, 1).repeat(1, 1, s)
        image[:, :, :s] = color_s
        image[:, :, -s:] = color_s

        return image


class Noise(TransformBase):
    NAME = "noise"
    IS_RANDOM = True
    PARAMS = {
        "std": SequenceParameter(float, length=3, default=None),
    }

    def __init__(self, std: List[Float]):
        super().__init__()
        self.std = std

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        std = torch.Tensor(context(self.std)).to(image.device)
        noise = torch.randn(image.shape).to(image.device)
        return image + noise * std.reshape(3, 1, 1)


class Edge(TransformBase):
    NAME = "edge"
    PARAMS = {
        "kernel_size": SequenceParameter(int, length=2, default=[3, 3]),
        "sigma": SequenceParameter(float, length=2, null=True, default=None),
        "amount": SequenceParameter(float, length=3, default=[1., 1., 1.]),
    }

    def __init__(self, kernel_size: List[Int], sigma: List[float], amount: List[float]):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        kernel_size = context(self.kernel_size)
        if self.sigma is None:
            sigma = None
        else:
            sigma = context(self.sigma)
        amount = torch.Tensor(context(self.amount)).to(image.device)
        edge = VF.gaussian_blur(image, kernel_size, sigma)
        edge = torch.clamp((image - edge) * amount, 0, 1)
        return edge


class Rotate(TransformBase):
    NAME = "rotate"
    PARAMS = {
        "degree": Parameter(float, default=None),
        "center": SequenceParameter(float, length=2, default=[0.5, 0.5]),
    }

    def __init__(self, degree: Float, center: List[Float]):
        super().__init__()
        self.degree = degree
        self.center = center

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        degree = context(self.degree)
        center = context(self.center)
        center_pix = [
            int(center[0] * image.shape[-1]),
            int(center[1] * image.shape[-2]),
        ]
        return VF.rotate(image, degree, center=center_pix)


class RandomRotate(TransformBase):
    NAME = "random_rotate"
    IS_RANDOM = True
    PARAMS = {
        "degree": SequenceParameter(float, length=2, default=[-1, 1]),
        "center": SequenceParameter(float, length=2, default=[0.5, 0.5]),
    }

    def __init__(self, degree: List[Float], center: List[Float]):
        super().__init__()
        self.degree = degree
        self.center = center

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        angle_min, angle_max = context(self.degree)
        center = context(self.center)
        angle = random.uniform(angle_min, angle_max)
        center_pix = [
            int(center[0] * image.shape[-1]),
            int(center[1] * image.shape[-2]),
        ]
        return VF.rotate(image, angle, center=center_pix)


class RandomScale(TransformBase):
    NAME = "random_scale"
    PARAMS = {
        "scale": SequenceParameter(float, length=2, default=None),
    }

    def __init__(self, scale: List[Float]):
        super().__init__()
        self.scale = scale

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        scale = context(self.scale)
        return VT.RandomAffine(degrees=0, scale=scale, fillcolor=None)(image)


class RandomTranslate(TransformBase):
    NAME = "random_translate"
    PARAMS = {
        "offset": SequenceParameter(float, length=2, default=None),
    }

    def __init__(self, offset: List[Float]):
        super().__init__()
        self.offset = offset

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        offset = context(self.offset)
        return VT.RandomAffine(degrees=0, translate=offset, fillcolor=None)(image)


class Shift(TransformBase):
    NAME = "shift"
    PARAMS = {
        "offset": SequenceParameter(float, length=2, default=None),
    }

    def __init__(self, offset: List[Float]):
        super().__init__()
        self.offset = offset

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        x, y = context(self.offset)
        return self._shift(image, x, y)

    def _shift(self, image: torch.Tensor, x: Union[int, float], y: Union[int, float]) -> torch.Tensor:
        if abs(x) < 1:
            x = x * image.shape[-1]
        if abs(y) < 1:
            y = y * image.shape[-2]
        x = int(x) % image.shape[-1]
        y = int(y) % image.shape[-2]

        if x != 0:
            image = torch.cat([image[:, :, -x:], image[:, :, :-x]], -1)

        if y != 0:
            image = torch.cat([image[:, -y:, :], image[:, :-y, :]], -2)

        return image


class RandomShift(Shift):
    NAME = "random_shift"
    PARAMS = {
        "offset": SequenceParameter(float, length=2, default=None),
    }

    def __init__(self, offset: List[Float]):
        super().__init__(offset)

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        mi, ma = context(self.offset)
        x = random.uniform(mi, ma)
        y = random.uniform(mi, ma)
        return self._shift(image, x, y)


class Add(TransformBase):
    NAME = "add"
    PARAMS = {
        "color": SequenceParameter(float, length=3, default=None),
    }

    def __init__(self, color: List[float]):
        super().__init__()
        self.color = color

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        color = torch.Tensor(context(self.color)).to(image.device).reshape(3, -1)
        return (image.reshape(3, -1) + color).reshape(image.shape)


class Multiply(TransformBase):
    NAME = "mul"
    PARAMS = {
        "color": SequenceParameter(float, length=3, default=None),
    }

    def __init__(self, color: List[float]):
        super().__init__()
        self.color = color

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        color = torch.Tensor(context(self.color)).to(image.device).reshape(3, -1)
        return (image.reshape(3, -1) * color).reshape(image.shape)


class Clamp(TransformBase):
    NAME = "clamp"
    PARAMS = {
        "range": SequenceParameter(float, length=2, default=None),
    }

    def __init__(self, range: List[float]):
        super().__init__()
        self.range = range

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        range = context(self.range)
        return torch.clamp(image, range[0], range[1])
