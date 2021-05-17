from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torchvision.transforms as VT
import torchvision.transforms.functional as VF

from .expression import Expression, ExpressionContext
from .parameters import Parameter, SequenceParameter, FrameTimeParameter
from .strings import value_str


Int = Union[int, Expression]
Float = Union[float, Expression]


constraints = dict()


class ConstraintBase(torch.nn.Module):
    """
    So, constraint modules must implement a forward function
    which takes the image tensor and an ExpressionContext
    and which must return a scalar tensor with the loss.

    The constructor must accept all defined parameters
    except the `weight`, `start` and `end` parameters,
    which are handled by ImageTrainer.
    """
    NAME = None
    PARAMS = None

    OUTER_PARAMS = ("weight", "start", "end")

    def __init_subclass__(cls, **kwargs):
        if cls.NAME is not None:
            constraints[cls.NAME] = cls

        if cls.PARAMS is not None:
            cls.PARAMS = {
                **cls.PARAMS,
                "weight": Parameter(
                    float, default=1.,
                    doc="""
                    A multiplier for the resulting loss value of the constraint.  
                    """
                ),
                "start": FrameTimeParameter(
                    default=0.,
                    doc="Start frame of the constraints. The constraint is inactive before this time."
                ),
                "end": FrameTimeParameter(
                    default=1.,
                    doc="End frame of the constraints. The constraint is inactive after this time."
                ),
                "loss": Parameter(
                    str, default="l2",
                    doc="""
                    The [loss function](https://en.wikipedia.org/wiki/Loss_function) 
                    used to calculate the difference (or error) between current and desired 
                    image.
                    
                    - `l1` or `mae`: [Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
                      is the mean of the absolute difference of each vector variable.
                    - `l2` or `mse`: [Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
                      is the mean of the squared difference of each vector variable. Compared to 
                      *mean absolute error*, it produces a smaller loss for small differences and 
                      a larger loss for large differences.
                    """
                ),
            }

    def __init__(self, loss: Union[str, Expression]):
        super().__init__()
        self.loss = loss

    @classmethod
    def strip_parameters(cls, params: dict):
        params = params.copy()
        for name in cls.OUTER_PARAMS:
            params.pop(name, None)
        return params

    def loss_function(
            self,
            value: torch.Tensor,
            target: torch.Tensor,
            context: ExpressionContext,
    ) -> torch.Tensor:
        loss = context(self.loss)
        if loss in ("l1", "mae"):
            return F.l1_loss(value, target)

        elif loss in ("l2", "mse"):
            return F.mse_loss(value, target)

        raise ValueError(f"Invalid loss function '{loss}' for constraint")

    def forward(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        raise NotImplementedError

    def description(self, context: ExpressionContext) -> str:
        if not self.PARAMS:
            return f"{self.NAME}()"

        param_strs = []
        for name, param in self.PARAMS.items():
            if name in self.OUTER_PARAMS:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            value = context(value)
            param_strs.append(f"{name}={value_str(value)}")

        param_strs = ", ".join(param_strs)
        return f"{self.NAME}({param_strs})"


class AboveBelowConstraintBase(ConstraintBase):

    WEIGHT_FACTOR = 1.

    PARAMS = {
        "above": SequenceParameter(
            float, length=3, default=None,
            doc="""
            If specified, the training loss increases if the current value is
            below the `above` value.  
            """
        ),
        "below": SequenceParameter(
            float, length=3, default=None,
            doc="""
            If specified, the training loss increases if the current value is
            above the `below` value.  
            """
        ),
    }

    def __init__(
            self,
            above: List[Union[float, Expression]],
            below: List[Union[float, Expression]],
            loss: str,
    ):
        super().__init__(loss=loss)
        assert above is not None or below is not None, "Must specify at least one of 'above' and 'below'"
        self.below = below
        self.above = above

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        raise NotImplementedError

    def forward(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        value = self.get_image_value(image, context)

        loss_sum = torch.tensor(0)

        if self.above is not None:
            target = torch.tensor(context(self.above)).to(image.device)
            loss = self.loss_function(target - value, torch.zeros(3).to(image.device), context)
            loss_sum = loss_sum + loss

        if self.below is not None:
            target = torch.tensor(context(self.below)).to(image.device)
            loss = self.loss_function(value - target, torch.zeros(3).to(image.device), context)
            loss_sum = loss_sum + loss

        return loss_sum * self.WEIGHT_FACTOR


class MeanConstraint(AboveBelowConstraintBase):
    """
    Pushes the image color mean above or below a threshold value
    """
    NAME = "mean"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        image = image.reshape(3, -1)
        return image.mean(-1)


class StdConstraint(AboveBelowConstraintBase):
    """
    Pushes the [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)
    above or below a threshold value.
    """
    NAME = "std"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        image = image.reshape(3, -1)
        return image.std(-1)


class SaturationConstraint(AboveBelowConstraintBase):
    """
    Pushes the saturation above or below a threshold value.

    The saturation is currently calculated as the difference of each
    color channel to the mean of all channels.
    """
    NAME = "saturation"
    WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        return get_mean_saturation(image)


def get_mean_saturation(image: torch.Tensor) -> torch.Tensor:
    color_planes = image.reshape(3, -1)
    mean_plane = color_planes.mean(0)
    return torch.abs(mean_plane.repeat(3, 1) - color_planes).mean(1)


class BlurConstraint(ConstraintBase):
    """
    Adds the difference between the image and a blurred version to
    the training loss.

    This is much more helpful than using the gaussian blur
    as a [post-processing](#postproc) step. When added to the
    training loss, the blurring keeps in balance with the
    actual image creation.

    Areas that CLIP is *excited about* will be constantly
    updated and will stand out of the blur, while *unexciting*
    areas get blurred a lot.
    """
    NAME = "blur"
    PARAMS = {
        "kernel_size": SequenceParameter(
            int, length=2, default=[3, 3],
            doc="""
            The size of the pixel window. Must be an **odd*, **positive** integer. 
            
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

    def __init__(
            self,
            kernel_size: List[Union[int, Expression]],
            sigma: List[Union[float, Expression]],
            loss: Union[str, Expression],
    ):
        super().__init__(loss=loss)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, image: torch.Tensor, context: ExpressionContext):
        blurred_image = get_expression_blur(image, self.kernel_size, self.sigma, context)

        loss = self.loss_function(
            image.reshape(3, -1),
            blurred_image.reshape(3, -1),
            context,
        )

        return 100. * loss


class EdgeMeanConstraint(AboveBelowConstraintBase):
    """
    Adds the difference between the current image and
    and an edge-detected version to the training constraint.

    A gaussian blur is used to detect the edges:

        edge = amount * abs(image - blur(image))

    """
    NAME = "edge_mean"
    WEIGHT_FACTOR = 100.

    PARAMS = {
        **AboveBelowConstraintBase.PARAMS,
        "kernel_size": SequenceParameter(
            int, length=2, default=[3, 3],
            doc="""
            The size of the pixel window of the gaussian blur. 
            Must be an **odd*, **positive** integer. 
            
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

    def __init__(
            self,
            above: List[Union[float, Expression]],
            below: List[Union[float, Expression]],
            kernel_size: List[Union[int, Expression]],
            sigma: List[Union[float, Expression]],
            loss: str,
    ):
        super().__init__(above=above, below=below, loss=loss)
        self.kernel_size = kernel_size
        self.sigma = sigma

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        blurred_image = get_expression_blur(image, self.kernel_size, self.sigma, context)
        edges = torch.abs(blurred_image - image)
        return edges.reshape(3, -1).mean(1)


def get_expression_blur(
        image: torch.Tensor,
        kernel_size: List[int],
        sigma: Optional[List[float]],
        context: ExpressionContext,
) -> torch.Tensor:
    kernel_size = context(kernel_size)
    kernel_size = [
        max(1, k+1 if k % 2 == 0 else k)
        for k in kernel_size
    ]
    if sigma is not None:
        sigma = [max(0.0001, s) for s in context(sigma)]

    return VF.gaussian_blur(image, kernel_size, sigma)


def get_edge_mean(image: torch.Tensor) -> torch.Tensor:
    blurred_image = VF.gaussian_blur(image, [3, 3], None)
    edges = (blurred_image - image)
    edges = torch.abs(edges)
    return edges.reshape(3, -1).mean(1)


class BorderConstraint(ConstraintBase):
    """
    Adds a border with a specific size and color to the training loss.
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

    def __init__(self, size: List[Int], color: List[Float], loss: Union[str, Expression],):
        super().__init__(loss=loss)
        self.size = size
        self.color = color

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
            image, image_border, context
        )


class NormalizeConstraint(ConstraintBase):
    """
    Adds image normalization to the training loss.

    The normalized version is found by moving the image colors
    into the range of [min](#targetsconstraintsnormalizemin)
    and [max](#targetsconstraintsnormalizemax).
    """
    NAME = "normalize"
    PARAMS = {
        "min": SequenceParameter(
            float, length=3, default=[0., 0., 0.],
            doc="""
            The desired lowest value in the image. 
            
            One color for gray-scale, three colors for **red**, **green** and **blue**.
            """
        ),
        "max": SequenceParameter(
            float, length=3, default=[1., 1., 1.],
            doc="""
            The desired highest value in the image. 
            
            One color for gray-scale, three colors for **red**, **green** and **blue**.
            """
        ),
    }

    def __init__(self, min: List[Float], max: List[Float], loss: Union[str, Expression]):
        super().__init__(loss=loss)
        self.min = min
        self.max = max

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
            image, normed_image, context
        )


class ContrastConstraint(AboveBelowConstraintBase):
    """
    Pushes the contrast above or below a threshold value.

    The contrast is currently calculated in the following way:

    The image pixels are divided into the ones that are
    above and below the pixel mean values. The contrast
    value is then the difference between the mean of the lower
    and the mean of the higher pixels.
    """
    NAME = "contrast"
    # WEIGHT_FACTOR = 100.

    def get_image_value(self, image: torch.Tensor, context: ExpressionContext):
        mean = image.mean(-1).mean(-1)
        mean = mean.reshape(3, 1, 1)

        low_mean = torch.cat([
            image[0][image[0] < mean[0]].mean(-1).unsqueeze(0),
            image[1][image[1] < mean[1]].mean(-1).unsqueeze(0),
            image[2][image[2] < mean[2]].mean(-1).unsqueeze(0),
        ])
        low_mean[torch.isnan(low_mean)] = mean.mean()

        high_mean = torch.cat([
            image[0][image[0] > mean[0]].mean(-1).unsqueeze(0),
            image[1][image[1] > mean[1]].mean(-1).unsqueeze(0),
            image[2][image[2] > mean[2]].mean(-1).unsqueeze(0),
        ])
        high_mean[torch.isnan(high_mean)] = mean.mean()

        # low_mean = image[image < mean].mean(-1).mean(-1)
        # high_mean = image[image > mean].mean(-1).mean(-1)
        if not low_mean.shape:
            low_mean = mean
        if not high_mean.shape:
            high_mean = mean

        spread = high_mean - low_mean
        return spread


class NoiseConstraint(ConstraintBase):
    """
    Adds the difference between the current image and
    a noisy image to the training loss.
    """
    NAME = "noise"

    PARAMS = {
        "std": SequenceParameter(
            float, length=3, default=None,
            doc="""
            Specifies the standard deviation of the noise distribution. 
            
            One value or three values to specify **red**, **green** and **blue** separately.
            """
        ),
    }

    def __init__(
            self,
            std: List[Union[Float, Expression]],
            loss: str,
    ):
        super().__init__(loss=loss)
        self.std = std

    def forward(self, image: torch.Tensor, context: ExpressionContext) -> torch.Tensor:
        std = torch.Tensor(context(self.std)).to(image.device).reshape(3, 1)
        image = image.reshape(3, -1)
        noisy_image = image + std * torch.randn(*image.shape).to(image.device)

        return 100. * self.loss_function(image, noisy_image, context)
