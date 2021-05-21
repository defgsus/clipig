import os
from pathlib import Path
import hashlib
import requests
from typing import List

import PIL.Image
import torch
import torchvision.transforms.functional as VT


CACHE_DIR = Path("~/.cache/img").expanduser()


def load_image(path: str) -> PIL.Image.Image:
    path = str(path)
    if path.startswith("http://") or path.startswith("https://"):
        return _load_web_image(path)

    return PIL.Image.open(path)


def load_image_tensor(path: str) -> torch.Tensor:
    return VT.to_tensor(load_image(path))


def _load_web_image(url: str) -> PIL.Image.Image:
    hash = hashlib.md5(url.encode("ascii")).hexdigest()

    if CACHE_DIR.joinpath(hash).exists():
        return PIL.Image.open(CACHE_DIR.joinpath(hash))

    response = requests.get(url)

    if not CACHE_DIR.exists():
        os.makedirs(CACHE_DIR)

    with open(CACHE_DIR.joinpath(hash), "wb") as fp:
        fp.write(response.content)

    return PIL.Image.open(CACHE_DIR.joinpath(hash))


def resize_crop(image: torch.Tensor, resolution: List[int]):
    width, height = image.shape[-1], image.shape[-2]

    if width != resolution[0] or height != resolution[1]:

        if width < height:
            factor = max(resolution) / width
        else:
            factor = max(resolution) / height

        image = VT.resize(
            image,
            [int(height * factor), int(width * factor)]
        )

        if image.shape[-2:] != resolution:
            image = VT.center_crop(image, resolution)
        
    return image
