import os
import sys
import hashlib
import argparse
import pathlib
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import clip
import PIL
from tqdm import tqdm

from src.clip_singleton import ClipSingleton
from src.images import load_image, resize_crop


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, nargs="+",
        help="Image files that should be rated via CLIP",
    )
    parser.add_argument(
        "-t", "--text", type=str, nargs="+",
        help="Text prompts to which images are compared",
    )
    parser.add_argument(
        "-i", "--image", type=str, nargs="+",
        help="Reference images to which images are compared",
    )
    parser.add_argument(
        "-tf", "--text-file", type=str, default=None,
        help="File with text prompts on each line to which images are compared",
    )
    parser.add_argument(
        "-if", "--image-file", type=str, default=None,
        help="File with reference image filenames on each line to which images are compared",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="auto",
        help="Device to run on, either 'auto', 'cuda' or 'cuda:1', etc... "
             "Default is 'auto'",
    )
    parser.add_argument(
        "-m", "--model", type=str, default="ViT-B/32",
        help=f"CLIP model to run, choices are {clip.available_models()}",
    )
    parser.add_argument(
        "-s", "--sort", type=int, default=None,
        help=f"Sort by text column, 1 for first, 2 for second, -1 for first descending, aso..",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help=f"Filename of a CSV file to store the output",
    )
    parser.add_argument(
        "-c", "--cache", type=bool, default=False, const=True, nargs="?",
        help=f"Enable caching of image features",
    )

    args = parser.parse_args()

    if not args.text and not args.image and not args.text_file and not args.image_file:
        print("Need to define at least one '-t/--text' or '-i/--image'")
        exit(1)

    return args


class ClipRater:

    CACHE_DIR = pathlib.Path("~").expanduser() / ".cache" / "clipig-rate"

    def __init__(
            self,
            filenames: List[str],
            texts: Optional[List[str]] = None,
            images: Optional[List[str]] = None,
            model: str = "ViT-B/32",
            device: str = "auto",
            batch_size: int = 10,
            caching: bool = False,
    ):
        self.filenames = filenames
        self.texts = texts or []
        self.images = images or []
        self.batch_size = batch_size
        self.caching = caching
        self.model, self.preprocess = ClipSingleton.get(model, device)
        self.device = self.model.logit_scale.device

    def rate(self) -> pd.DataFrame:
        image_features, filenames = self._get_image_file_features(self.filenames, "encoding images")
        compare_features, columns = self._get_all_compare_features()
        compare_features = compare_features.type_as(image_features)

        similarities = image_features @ compare_features.T * 100.

        df = pd.DataFrame(similarities.numpy(), index=filenames, columns=columns)
        return df

    def _get_image_features(self, images: List) -> torch.Tensor:
        with torch.no_grad():
            images = torch.cat(images).to(self.device)
            image_features = self.model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.cpu()

    def _get_text_features(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_tokens = [
                clip.tokenize(text)
                for text in texts
            ]
            text_tokens = torch.cat(text_tokens).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu()

    def _get_image_file_features(
            self,
            filenames: List[str],
            desc: str = None
    ) -> Tuple[torch.Tensor, List[str]]:
        image_features = dict()
        images = []
        for filename in tqdm(filenames, desc=desc):
            if self.caching:
                feature = self._get_cache(filename)
                if feature is not None:
                    image_features[filename] = feature
                    continue

            try:
                image = load_image(filename)
                image = resize_crop(image, [224, 224])
                image = self.preprocess(image)
            except (PIL.UnidentifiedImageError, OSError):
                print(f"{filename} seems to be no image", file=sys.stderr)
                continue

            images.append((filename, image))

            if len(images) >= self.batch_size:
                features = self._get_image_features([i[1].unsqueeze(0) for i in images])
                for (filename, _), feature in zip(images, features):
                    feature = feature
                    image_features[filename] = feature
                    if self.caching:
                        self._save_cache(filename, feature)
                images = []

        if images:
            features = self._get_image_features([i[1].unsqueeze(0) for i in images])
            for (filename, _), feature in zip(images, features):
                image_features[filename] = feature
                if self.caching:
                    self._save_cache(filename, feature)

        filenames = list(filter(lambda f: f in image_features, filenames))

        image_features = [
            image_features[fn].unsqueeze(0)
            for fn in filenames
        ]

        image_features = torch.cat(image_features)
        return image_features, filenames

    def _get_all_compare_features(self) -> Tuple[torch.Tensor, List[str]]:
        compare_features = []
        compare_feature_names = []
        texts = self.texts
        while texts:
            compare_features.append(self._get_text_features(texts[:self.batch_size]))
            compare_feature_names += texts[:self.batch_size]
            texts = texts[self.batch_size:]

        if self.images:
            features, filenames = self._get_image_file_features(self.images, "encoding compare images")
            compare_features.append(features)
            compare_feature_names += filenames

        return torch.cat(compare_features), compare_feature_names

    def _get_cache_filename(self, filename: str) -> pathlib.Path:
        if filename.startswith("https://") or filename.startswith("http://"):
            key = filename
        else:
            path = pathlib.Path(filename).resolve()
            key = f"{path}-{path.stat().st_mtime}"
        return self.CACHE_DIR / hashlib.md5(key.encode("utf-8")).hexdigest()

    def _get_cache(self, filename: str) -> Optional[torch.Tensor]:
        fn = self._get_cache_filename(filename)
        if fn.exists():
            return torch.load(fn)

    def _save_cache(self, filename: str, tensor: torch.Tensor):
        fn = self._get_cache_filename(filename)
        if not self.CACHE_DIR.exists():
            os.makedirs(self.CACHE_DIR)
        torch.save(tensor, str(fn))


def trim_length(s: str, max_len: int, front: Union[bool, str] = False) -> str:
    if front == "auto":
        front = pathlib.Path(s).exists()

    if len(s) > max_len:
        if front:
            return ".." + s[-max_len+2:]
        return s[:max_len-2] + ".."
    return s


def dump_similarities(similarities: pd.DataFrame):
    similarities = similarities.copy()

    filenames = list(similarities.index)
    if len(filenames) > 1:
        path = str(pathlib.Path(filenames[0]).parent) + os.path.sep
        if all(f.startswith(path) for f in filenames):
            filenames = [f[len(path):] for f in filenames]

    similarities.columns = [trim_length(i, 20, "auto") for i in similarities.columns]
    similarities.index = [trim_length(i, 30, "auto") for i in filenames]

    # similarities = similarities.round(2)
    print(similarities)

    print(similarities.describe())


def main():
    args = parse_arguments()

    texts = args.text or []
    images = args.image or []

    if args.text_file:
        texts = texts + list(
            filter(bool, pathlib.Path(args.text_file).read_text().splitlines())
        )

    if args.image_file:
        images = images + list(
            filter(bool, pathlib.Path(args.image_file).read_text().splitlines())
        )

    rater = ClipRater(
        filenames=args.file,
        texts=texts,
        images=images,
        model=args.model,
        device=args.device,
        caching=args.cache,
    )
    similarities = rater.rate()

    if args.sort:
        similarities.sort_values(
            by=similarities.columns[abs(args.sort)-1],
            ascending=args.sort > 0,
            inplace=True,
        )
    else:
        similarities.sort_index(inplace=True)

    dump_similarities(
        similarities,
    )

    if args.output:
        similarities.to_csv(args.output)


if __name__ == "__main__":
    main()
