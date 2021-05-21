from typing import Union, Sequence, Type, Tuple, Optional, List

import numpy as np
import torch
import torch.nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image
import PIL.Image

from .images import load_image_tensor


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

    def initialize(self, parameters: dict):
        mean = torch.Tensor(parameters["mean"])
        std = torch.Tensor(parameters["std"])

        img = None
        if parameters["image_tensor"]:
            img = torch.Tensor(parameters["image_tensor"])
        elif parameters["image"]:
            img = load_image_tensor(parameters["image"])

        if img is not None:
            if img.shape != self.pixels.shape:
                scale = min(img.shape[1:]) / min(self.resolution)
                img = VF.resize(img, [int(img.shape[2] / scale), int(img.shape[1] / scale)])
                img = VF.center_crop(img, self.resolution)
            pixels = img

            if not parameters["image_tensor"]:
                pixels = pixels * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)
                # pixels = VF.normalize(pixels, mean, std)
        else:
            pixels = torch.randn((3, self.resolution[1], self.resolution[0]))
            pixels = pixels * std.reshape(-1, 1, 1) + mean.reshape(-1, 1, 1)

        if not parameters["image_tensor"]:
            pixels = torch.clamp(pixels, 0, 1)

        with torch.no_grad():
            self.pixels[...] = pixels

    def resize(self, resolution: List[int]):
        self.resolution = list(resolution)
        self.pixels = torch.nn.Parameter(
            VF.resize(self.pixels, resolution)
        )

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
