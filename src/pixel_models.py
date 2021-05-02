from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image
import PIL.Image


class PixelsBase(torch.nn.Module):

    def __init__(
            self,
            resolution: Sequence[int]
    ):
        super().__init__()
        self.resolution = list(resolution)

    def save_image(self, filename: str):
        save_image(self.forward(), filename)

    def info_str(self) -> str:
        raise NotImplementedError

    def mean_saturation(self) -> torch.Tensor:
        raise NotImplementedError

    def initialize(self, parameters: dict):
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

    def initialize(self, parameters: dict):
        pixels = torch.randn((3, self.resolution[1] * self.resolution[0]))

        mean = torch.Tensor(parameters["mean"]).reshape(-1, 1)
        std = torch.Tensor(parameters["std"]).reshape(-1, 1)
        pixels = pixels * std + mean

        pixels = pixels.reshape(3, self.resolution[1], self.resolution[0])

        if parameters["image"]:
            img = VF.to_tensor(PIL.Image.open(parameters["image"]))
            if img.shape != pixels.shape:
                scale = min(img.shape[1:]) / min(self.resolution)
                img = VF.resize(img, [int(img.shape[2] / scale), int(img.shape[1] / scale)])
                img = VF.center_crop(img, self.resolution)
            pixels = pixels + img

        pixels = torch.clamp(pixels, 0, 1)

        with torch.no_grad():
            self.pixels[...] = pixels

    def info_str(self) -> str:
        return f"mean rgbs " \
               f"{float(self.pixels[0].mean()):.3f}, " \
               f"{float(self.pixels[1].mean()):.3f}, " \
               f"{float(self.pixels[2].mean()):.3f}, " \
               f"{float(self.mean_saturation()):.3f}"

    def forward(self):
        return torch.clamp(self.pixels, 0, 1)

    def mean_saturation(self) -> torch.Tensor:
        color_planes = self.pixels.reshape(3, -1)
        mean_plane = color_planes.mean(dim=0, keepdim=True)
        saturation_plane = torch.abs(mean_plane.repeat(3, 1) - color_planes).sum(0, keepdim=True) / 3.
        return saturation_plane.mean()

    def blur(
            self,
            kernel_size: int = 3,
            sigma: Union[float, Tuple[float]] = 0.35,
    ):
        with torch.no_grad():
            pixels = self.pixels
            blurred_pixels = VF.gaussian_blur(pixels, [kernel_size, kernel_size], [sigma, sigma])
            self.pixels[...] = blurred_pixels

    def add(
            self,
            rgb: Sequence[float],
    ):
        assert len(rgb) == 3, f"Expected sequence of 3 floats, got {rgb}"

        with torch.no_grad():
            rgb = torch.Tensor(rgb).to(self.pixels.device).reshape(3, -1)
            self.pixels[...] = (
                    self.pixels.reshape(3, -1) + rgb
            ).reshape(3, self.resolution[1], self.resolution[0])

    def multiply(
            self,
            rgb: Sequence[float],
    ):
        assert len(rgb) == 3, f"Expected sequence of 3 floats, got {rgb}"

        with torch.no_grad():
            rgb = torch.Tensor(rgb).to(self.pixels.device).reshape(3, -1)
            self.pixels[...] = (
                    self.pixels.reshape(3, -1) * rgb
            ).reshape(3, self.resolution[1], self.resolution[0])
