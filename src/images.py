import os
from pathlib import Path
import hashlib
import requests

import PIL.Image
import torch
from torchvision.transforms.functional import to_tensor


CACHE_DIR = Path("~/.cache/img").expanduser()


def load_image(path: str) -> PIL.Image.Image:
    path = str(path)
    if path.startswith("http://") or path.startswith("https://"):
        return _load_web_image(path)

    return PIL.Image.open(path)


def load_image_tensor(path: str) -> torch.Tensor:
    return to_tensor(load_image(path))


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
