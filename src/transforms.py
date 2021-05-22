import random
from typing import Union, Sequence, List, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torch.fft as fft
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid

from .expression import Expression, ExpressionContext
from .parameters import (
    Parameter, SequenceParameter, FrameTimeParameter, _add_transforms_parameters
)

Int = Union[int, Expression]
Float = Union[float, Expression]
Str = Union[str, Expression]


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

        # TODO: would be nice to have start/end on each transform
        #   but most transforms only have a single parameter so i'd
        #   like to support both notations, e.g.:
        #       resize: 224
        #    or
        #       resize:
        #         size: 224
        #         start: 10%
        # cls.PARAMS = {
        #     **cls.PARAMS,
        #     "start": FrameTimeParameter(
        #         default=0.,
        #         doc="Start frame of the transform. The transform is inactive before this time."
        #     ),
        #     "end": FrameTimeParameter(
        #         default=1.,
        #         doc="End frame of the transform. The transform is inactive after this time."
        #     ),
        # }

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        raise NotImplementedError


class Blur(TransformBase):
    """
    A gaussian blur is applied to the pixels.

    See [torchvision gaussian_blur](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.gaussian_blur).
    """
    NAME = "blur"
    PARAMS = {
        "kernel_size": SequenceParameter(
            int, length=2, default=[3, 3],
            doc="""
            The size of the pixel window. Must be an **odd**, **positive** integer. 
            
            Two numbers define **width** and **height** separately.
            """
        ),
        "sigma": SequenceParameter(
            float, length=2, null=True, default=None,
            doc="""
            Gaussian kernel standard deviation. The larger, the more *blurry*.
            
            If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.
            
            Two numbers define sigma for **x** and **y** separately.
            """
        ),
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
    """
    The resolution of the image is changed.
    """
    NAME = "resize"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(
            int, length=2, default=None,
            doc="""
            One integer for square images, two numbers to specify **width** and **height**. 
            """
        ),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VF.resize(image, [size[1], size[0]])


class CenterCrop(TransformBase):
    """
    Crops an image of the given resolution from the center.

    See [torchvision center_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.center_crop).
    """
    NAME = "center_crop"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(
            int, length=2, default=None,
            doc="""
            One integer for square images, two numbers to specify **width** and **height**. 
            """
        ),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VF.center_crop(image, size)


class RandomCrop(TransformBase):
    """
    Crops a section of the specified resolution from a random position in the image.

    See [torchvision random_crop](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.functional.random_crop)
    """
    NAME = "random_crop"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(
            int, length=2, default=None,
            doc="""
            One integer for square images, two numbers to specify **width** and **height**. 
            """
        ),
    }

    def __init__(self, size: List[Int]):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        size = context(self.size)
        return VT.RandomCrop(size=size)(image)


class Crop(TransformBase):
    """
    Crops a specified section from the image.
    """
    NAME = "crop"
    IS_RESIZE = True
    PARAMS = {
        "xywh": SequenceParameter(
            float, length=4, default=None,
            doc="""
            4 numbers: **x** and **y** of top-left corner followed by **width** and **height**.
            
            A number between 0 and 1 is considered a fraction of the full resolution.
            A number greater or equal to 1 is considered a pixel coordinate 
            """
        ),
    }

    def __init__(self, xywh: List[Int]):
        super().__init__()
        self.xywh = xywh

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        x, y, w, h = context(self.xywh)
        if x < 1:
            x = x * image.shape[-1]
        if y < 1:
            y = y * image.shape[-2]
        if w < 1:
            w = w * image.shape[-1]
        if h < 1:
            h = h * image.shape[-2]
        return VF.crop(image, int(y), int(x), int(h), int(w))


class Repeat(TransformBase):
    """
    Repeats the image a number of times in the right and bottom direction.
    """
    NAME = "repeat"
    IS_RESIZE = True
    PARAMS = {
        "count": SequenceParameter(
            int, length=2, default=None,
            doc="""
            One integer to specify **x** and **y** at the same time, 
            or two integers to specify them separately.  
            """
        ),
    }

    def __init__(self, count: List[Int]):
        super().__init__()
        self.count = count

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        count = context(self.count)
        return image.repeat(1, 1, count[0]).repeat(1, count[1], 1)


class Pad(TransformBase):
    """
    Pads the image with additional pixels at the borders.
    """
    NAME = "pad"
    IS_RESIZE = True
    PARAMS = {
        "size": SequenceParameter(
            int, length=2, default=None,
            doc="""
            The number of columns/rows to add. 
            
            One integer to specify **x** and **y** at the same time, 
            or two integers to specify them separately.
            
            E.g. `1, 2` would add 1 column left and one column right of
            the image and two rows on top and bottom respectively.
            """
        ),
        "color": SequenceParameter(
            float, length=3, default=[0., 0., 0.],
            doc="""
            The color of the pixels that are padded around the image.
            """
        ),
        "mode": Parameter(
            str, default="fill",
            doc="""
            The way the padded area is filled.
            
            - `fill`: fills everything with the `color` value
            - `edge`: repeats the edge pixels
            """
        )
    }

    def __init__(self, size: List[Int], color: List[Float], mode: Str):
        super().__init__()
        self.size = size
        self.color = color
        self.mode = mode

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        mode = context(self.mode)
        size = [max(0, s) for s in context(self.size)]
        color = torch.Tensor(context(self.color)).to(image.device)

        if mode == "fill":
            if size[0]:
                column = color.reshape(3, 1, 1).repeat(1, image.shape[-2], size[0])
                image = torch.cat([column, image, column], dim=-1)

            if size[1]:
                row = color.reshape(3, 1, 1).repeat(1, size[1], image.shape[-1])
                image = torch.cat([row, image, row], dim=-2)

        elif mode == "edge":
            if size[0]:
                column_left = image[:, :, :1].repeat(1, 1, size[0])
                column_right = image[:, :, -1:].repeat(1, 1, size[0])
                image = torch.cat([column_left, image, column_right], dim=-1)

            if size[1]:
                row_top = image[:, :1, :].repeat(1, size[1], 1)
                row_bottom = image[:, -1:, :].repeat(1, size[1], 1)
                image = torch.cat([row_top, image, row_bottom], dim=-2)

        elif mode == "wrap":
            if size[0]:
                column_left = image[:, :, -size[0]:]
                column_right = image[:, :, :size[0]]
                image = torch.cat([column_left, image, column_right], dim=-1)

            if size[1]:
                row_top = image[:, -size[1]:, :]
                row_bottom = image[:, :size[1], :]
                image = torch.cat([row_top, image, row_bottom], dim=-2)

        return image


class Border(TransformBase):
    """
    Draws a border on the edge of the image. The resolution is not changed.
    """
    NAME = "border"
    PARAMS = {
        "size": SequenceParameter(
            int, length=2, default=[1, 1],
            doc="""
            One integer two specify **width** and **height** at the same time, 
            or two integers to specify them separately.  
            """
        ),
        "color": SequenceParameter(
            float, length=3, default=[0., 0., 0.],
            doc="""
            The color of the border as float numbers in the range `[0, 1]`.
            
            Three numbers for **red**, **green** and **blue** or a single number 
            to specify a gray-scale. 
            """
        ),
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
    """
    Adds noise to the image.

    The noise has a scalable normal distribution around zero.
    """
    NAME = "noise"
    IS_RANDOM = True
    PARAMS = {
        "std": SequenceParameter(
            float, length=3, default=None,
            doc="""
            Specifies the standard deviation of the noise distribution. 
            
            One value or three values to specify **red**, **green** and **blue** separately.
            """
        ),
    }

    def __init__(self, std: List[Float]):
        super().__init__()
        self.std = std

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        std = torch.Tensor(context(self.std)).to(image.device)
        noise = torch.randn(image.shape).to(image.device)
        return image + noise * std.reshape(3, 1, 1)


class FNoise(TransformBase):
    """
    Adds noise to the image's fourier space.

    It's just a bit different than the normal [noise](reference.md#targetstransformsnoise).

    The noise has a scalable normal distribution around zero.
    """
    NAME = "fnoise"
    IS_RANDOM = True
    PARAMS = {
        "std": SequenceParameter(
            float, length=3, default=None,
            doc="""
            Specifies the standard deviation of the noise distribution. 
            The actual value is multiplied by `15.0` to give a visually 
            similar distribution as the normal [noise](reference.md#targetstransformsnoise).
            
            One value or three values to specify **red**, **green** and **blue** separately.
            """
        ),
    }

    def __init__(self, std: List[Float]):
        super().__init__()
        self.std = std

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        std = 15. * torch.Tensor(context(self.std)).to(image.device).reshape(3, 1)

        space = fft.rfft(image.reshape(3, -1))
        space.real = space.real + torch.randn(space.shape).to(image.device) * std
        space.imag = space.imag + torch.randn(space.shape).to(image.device) * std

        return fft.irfft(space).reshape(*image.shape)


class Edge(TransformBase):
    """
    This removes everything except edges and generally has a bad effect on image
    quality. It might be useful, however...

    A gaussian blur is used to detect the edges:

        edge = amount * abs(image - blur(image))
    """
    NAME = "edge"
    PARAMS = {
        "kernel_size": SequenceParameter(
            int, length=2, default=[3, 3],
            doc="""
            The size of the pixel window used for gaussian blur. 
            Must be an **odd**, **positive** integer. 
            
            Two numbers define **width** and **height** separately.
            """
        ),
        "sigma": SequenceParameter(
            float, length=2, null=True, default=None,
            doc="""
            Gaussian kernel standard deviation. The larger, the more *blurry*.
            
            If not specified it will default to `0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8`.
            
            Two numbers define sigma for **x** and **y** separately.
            """
        ),
        "amount": SequenceParameter(
            float, length=3, default=[1., 1., 1.],
            doc="""
            A multiplier for the edge value. Three numbers to specify 
            **red**, **green** and **blue** separately.
            """
        ),
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
    """
    Rotates the image.

    The resolution is not changed and areas outside of the image
    are filled with black (zero).
    """
    NAME = "rotate"
    PARAMS = {
        "degree": Parameter(
            float, default=None,
            doc="""
            The counter-clockwise angle of ration in degrees (`[0, 360]`).
            """
        ),
        "center": SequenceParameter(
            float, length=2, default=[0.5, 0.5],
            doc="""
            The center of rotation in the range `[0, 1]`. 
            
            Two numbers to specify **x** and **y** separately.
            """
        ),
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
    """
    Randomly rotates the image.

    Degree and center of rotation are chosen randomly between in the range
    of the specified values.

    The resolution is not changed and areas outside of the image
    are filled with black (zero).
    """
    NAME = "random_rotate"
    IS_RANDOM = True
    PARAMS = {
        "degree": SequenceParameter(
            float, length=2, default=None,
            doc="""
            The minimum and maximum counter-clockwise angle of ration in degrees.
            """
        ),
        "center": SequenceParameter(
            float, length=2, default=[0.5, 0.5],
            doc="""
            The minimum and maximum center of rotation (for x and y) in the range `[0, 1]`. 
            """
        ),
    }

    def __init__(self, degree: List[Float], center: List[Float]):
        super().__init__()
        self.degree = degree
        self.center = center

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        angle_min, angle_max = context(self.degree)
        center_min, center_max = context(self.center)

        angle = random.uniform(angle_min, angle_max)
        center_x = random.uniform(center_min, center_max)
        center_y = random.uniform(center_min, center_max)

        center_pix = [
            int(center_x * image.shape[-1]),
            int(center_y * image.shape[-2]),
        ]
        return VF.rotate(image, angle, center=center_pix)


class RandomScale(TransformBase):
    """
    Randomly scales an image in the range specified.

    See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).

    The resolution does not change, only contents are scaled.
    Areas outside of the image are filled with black (zero).
    """
    NAME = "random_scale"
    PARAMS = {
        "scale": SequenceParameter(
            float, length=2, default=None,
            doc="""
            Minimum and maximum scale, where `0.5` means half and `2.0` means double.
            """
        ),
    }

    def __init__(self, scale: List[Float]):
        super().__init__()
        self.scale = scale

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        scale = context(self.scale)
        return VT.RandomAffine(degrees=0, scale=scale, fillcolor=None)(image)


class RandomTranslate(TransformBase):
    """
    Randomly translates an image in the specified range.

    The resolution does not change.
    Areas outside of the image are filled with black (zero).

    See [torchvision RandomAffine](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomAffine).
    """
    NAME = "random_translate"
    PARAMS = {
        "offset": SequenceParameter(
            float, length=2, default=None,
            doc="""
            Maximum absolute fraction for horizontal and vertical translations. 
            For example: `random_translate: a, b`, then horizontal shift is randomly sampled in 
            the range `-img_width * a < dx < img_width * a` and vertical shift is randomly sampled in the range 
            `-img_height * b < dy < img_height * b`.
            """
        ),
    }

    def __init__(self, offset: List[Float]):
        super().__init__()
        self.offset = offset

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        offset = context(self.offset)
        return VT.RandomAffine(degrees=0, translate=offset, fillcolor=None)(image)


class Shift(TransformBase):
    """
    This translates the pixels of the image.

    Pixels that are moved outside get attached on the other side.
    """
    NAME = "shift"
    PARAMS = {
        "offset": SequenceParameter(
            float, length=2, default=None,
            doc="""
            A number **larger 1** or **smaller -1** translates by the actual pixels.
            
            A number **between -1 and 1** translates by the fraction of the image resolution.
            E.g., `shift: .5` would move the center of the image to the previous bottom-right
            corner.  
            
            A single number specifies translation on both **x** and **y** axes while
            two numbers specify them separately.
            """
        ),
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
    """
    This randomly translates the pixels of the image.

    Pixels that are moved outside get attached on the other side.
    """
    NAME = "random_shift"
    PARAMS = {
        "offset": SequenceParameter(
            float, length=2, default=None,
            doc="""
            Specifies the random range of translation.
            
            A number **larger 1** or **smaller -1** translates by the actual pixels.
            
            A number **between -1 and 1** translates by the fraction of the image resolution.
            E.g., `shift: 0 1` would randomly translate the image to every possible position
            given it's resolution.
            """
        ),
    }

    def __init__(self, offset: List[Float]):
        super().__init__(offset)

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        mi, ma = context(self.offset)
        x = random.uniform(mi, ma)
        y = random.uniform(mi, ma)
        return self._shift(image, x, y)


class Add(TransformBase):
    """
    Adds a fixed value to all pixels.
    """
    NAME = "add"
    PARAMS = {
        "color": SequenceParameter(
            float, length=3, default=None,
            doc="""
            Three numbers specify **red**, **green** and **blue** while a 
            single number specifies a gray-scale color.
            """
        ),
    }

    def __init__(self, color: List[float]):
        super().__init__()
        self.color = color

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        color = torch.Tensor(context(self.color)).to(image.device).reshape(3, -1)
        return (image.reshape(3, -1) + color).reshape(image.shape)


class Multiply(TransformBase):
    """
    Multiplies all pixels by a fixed value.
    """
    NAME = "mul"
    PARAMS = {
        "color": SequenceParameter(
            float, length=3, default=None,
            doc="""
            Three numbers specify **red**, **green** and **blue** while a 
            single number specifies a gray-scale color.
            """
        ),
    }

    def __init__(self, color: List[float]):
        super().__init__()
        self.color = color

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        color = torch.Tensor(context(self.color)).to(image.device).reshape(3, -1)
        return (image.reshape(3, -1) * color).reshape(image.shape)


class Clamp(TransformBase):
    """
    Clamps the pixels into a fixed range.
    """
    NAME = "clamp"
    PARAMS = {
        "range": SequenceParameter(
            float, length=2, default=None,
            doc="""
            First number is the minimum allowed value for all color channels, 
            second is the maximum allowed value.
            
            An image displayed on screen or converted to a file does only include
            values in the range of `[0, 1]`.
            """
        ),
    }

    def __init__(self, range: List[float]):
        super().__init__()
        self.range = range

    def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        range = context(self.range)
        return torch.clamp(image, range[0], range[1])



# TODO: not really helpful a.t.m.
#
# class Fft(TransformBase):
#     """
#     Fourier transform
#     """
#     NAME = "fft"
#     PARAMS = {
#         "real": Parameter(
#             bool, default=None,
#             doc="""
#             Return real or imaginary part
#             """
#         ),
#     }
#
#     def __init__(self, real: Union[bool, Expression]):
#         super().__init__()
#         self.real = real
#
#     def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
#         space = fft.rfft(image)
#         return torch.cat([space.real, space.imag], dim=-1)
#         if context(self.real):
#             return f_space.real
#         else:
#             return f_space.imag
#
#
# class IFft(TransformBase):
#     """
#     Fourier transform
#     """
#     NAME = "ifft"
#     PARAMS = {
#         "real": Parameter(
#             bool, default=None,
#             doc="""
#             Return real or imaginary part
#             """
#         ),
#     }
#
#     def __init__(self, real: Union[bool, Expression]):
#         super().__init__()
#         self.real = real
#
#     def __call__(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
#         space = torch.complex(
#             image[..., :image.shape[-1] // 2],
#             image[..., image.shape[-1] // 2:],
#         )
#         return fft.irfft(space)
#         #if context(self.real):
#         #    return f_space.real
#         #else:
#         #    return f_space.imag


_add_transforms_parameters(transformations)
