import os
from pathlib import Path
import hashlib
import requests
from typing import List, Union

import PIL.Image
import torch
import torchvision.transforms.functional as VT


CACHE_DIR = Path("~/.cache/img").expanduser()


INTERPOLATIONS = {
    "nearest": PIL.Image.NEAREST,
    "linear": PIL.Image.BILINEAR,
    "cubic": PIL.Image.BICUBIC,
}


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


def resize_crop(
        image: Union[torch.Tensor, PIL.Image.Image],
        resolution: List[int],
) -> Union[torch.Tensor, PIL.Image.Image]:
    if isinstance(image, PIL.Image.Image):
        width, height = image.width, image.height
    else:
        width, height = image.shape[-1], image.shape[-2]

    if width != resolution[0] or height != resolution[1]:

        if width < height:
            factor = max(resolution) / width
        else:
            factor = max(resolution) / height

        image = VT.resize(
            image,
            [int(height * factor), int(width * factor)],
            interpolation=PIL.Image.BICUBIC,
        )

        if isinstance(image, PIL.Image.Image):
            width, height = image.width, image.height
        else:
            width, height = image.shape[-1], image.shape[-2]

        if width != resolution[0] or height != resolution[1]:
            image = VT.center_crop(image, resolution)
        
    return image


def get_interpolation(name: str) -> int:
    """
    Convert string to PIL interpolation
    :param name: str
    :return: int, like PIL.Image.BICUBIC
    """
    if name in INTERPOLATIONS:
        return INTERPOLATIONS[name]

    inters = ", ".join(f"'{i}'" for i in INTERPOLATIONS)
    raise ValueError(
        f"Interpolation must be one of {inters}, got '{name}'"
    )
