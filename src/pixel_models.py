from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image
import PIL.Image

from .images import load_image_tensor, resize_crop


class PixelsBase(torch.nn.Module):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__()
        self.resolution = list(resolution)

    def set_pixels(self, pixels: torch.Tensor):
        raise NotImplementedError

    def get_pixels(self) -> torch.Tensor:
        return self.forward()

    def save_image(self, filename: str):
        save_image(self.get_pixels(), filename)

    def info_str(self) -> str:
        return f"mean rgbs " \
               f"{float(self.pixels[0].mean()):.3f}, " \
               f"{float(self.pixels[1].mean()):.3f}, " \
               f"{float(self.pixels[2].mean()):.3f}, " \
               f"{float(self.mean_saturation()):.3f}"

    def mean_saturation(self) -> torch.Tensor:
        color_planes = self.get_pixels().reshape(3, -1)
        mean_plane = color_planes.mean(dim=0, keepdim=True)
        saturation_plane = torch.abs(mean_plane.repeat(3, 1) - color_planes).sum(0, keepdim=True) / 3.
        return saturation_plane.mean()

    def initialize(
            self,
            mean: Sequence[float],
            std: Sequence[float],
            image: Optional[str] = None,
            image_tensor: Optional[Sequence] = None,
            resolution: Optional[Sequence[int]] = None,
    ):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        resolution = resolution or self.resolution

        img = None
        if image_tensor is not None:
            img = torch.Tensor(image_tensor)
            resolution = self.resolution
        elif image:
            img = load_image_tensor(image)

        if img is not None:
            pixels = resize_crop(img, resolution)

            if image_tensor is None:
                pixels = pixels * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
                # pixels = VF.normalize(pixels, mean, std)
        else:
            if resolution is None:
                pixels = torch.randn(3, resolution[1], resolution[0])
            else:
                pixels = torch.randn((3, resolution[1], resolution[0]))
            pixels = pixels * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)

        if resolution[0] != self.resolution[0] or resolution[1] != self.resolution[1]:
            pixels = VF.resize(pixels, self.resolution[::-1], PIL.Image.BICUBIC)

        if image_tensor is None:
            pixels = torch.clamp(pixels, 0, 1)

        with torch.no_grad():
            self.set_pixels(pixels)

    def resize(self, resolution: List[int]):
        raise NotImplementedError


class PixelsRGB(PixelsBase):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__(resolution)
        self.pixels = torch.nn.Parameter(
            torch.rand((3, resolution[1], resolution[0]))
        )

    def set_pixels(self, pixels: torch.Tensor):
        self.pixels[...] = pixels

    def resize(self, resolution: List[int], interpolation: int = PIL.Image.BICUBIC):
        self.resolution = list(resolution)
        self.pixels = torch.nn.Parameter(
            VF.resize(self.pixels, self.resolution[::-1], interpolation=interpolation)
        )

    def forward(self):
        return torch.clamp(self.pixels, 0, 1)
