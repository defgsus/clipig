import os
import argparse
import pathlib
from typing import List, Optional, Tuple, Union

import pandas as pd
import torch
import clip
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

    args = parser.parse_args()

    if not args.text and not args.image:
        print("Need to define at least one '-t/--text' or '-i/--image'")
        exit(1)

    return args


class ClipRater:

    def __init__(
            self,
            filenames: List[str],
            texts: List[str],
            images: List[str],
            model: str = "ViT-B/32",
            device: str = "auto",
            batch_size: int = 10,
    ):
        self.filenames = filenames
        self.texts = texts
        self.images = images
        self.batch_size = batch_size
        self.model, self.preprocess = ClipSingleton.get(model, device)
        self.device = self.model.logit_scale.device

    def rate(self) -> pd.DataFrame:
        image_features = self._get_image_file_features(self.filenames, "encoding images")
        compare_features, columns = self._get_all_compare_features()

        similarities = 100. * image_features @ compare_features.T

        df = pd.DataFrame(similarities.numpy(), index=self.filenames, columns=columns)
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

    def _get_image_file_features(self, filenames: List[str], desc: str = None) -> torch.Tensor:
        image_features = []
        images = []
        for filename in tqdm(filenames, desc=desc):
            image = load_image(filename)
            image = resize_crop(image, [224, 224])
            image = self.preprocess(image)
            images.append(image.unsqueeze(0))

            if len(images) >= self.batch_size:
                image_features.append(self._get_image_features(images))
                images = []

        if images:
            image_features.append(self._get_image_features(images))

        image_features = torch.cat(image_features)
        return image_features

    def _get_all_compare_features(self) -> Tuple[torch.Tensor, List[str]]:
        compare_features = []
        texts = self.texts
        while texts:
            compare_features.append(self._get_text_features(texts[:self.batch_size]))
            texts = texts[self.batch_size:]

        if self.images:
            compare_features.append(self._get_image_file_features(self.images, "encoding compare images"))

        return torch.cat(compare_features), (self.texts or []) + (self.images or [])


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


if __name__ == "__main__":

    args = parse_arguments()

    rater = ClipRater(
        filenames=args.file,
        texts=args.text,
        images=args.image,
        model=args.model,
        device=args.device,
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
