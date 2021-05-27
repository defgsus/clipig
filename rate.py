import argparse
import pathlib
from typing import List, Optional

import torch
import clip

from src.clip_singleton import ClipSingleton
from src.images import load_image, resize_crop


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+",
        help="Image files that should be rated via CLIP",
    )
    parser.add_argument(
        "-t", "--text", type=str, nargs="+",
        help="Text prompts which are used to rate the images",
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

    args = parser.parse_args()

    if not args.text:
        print("Need to define at least one '-t/--text'")
        exit(1)

    return args


def rate_images(
        filenames: List[str],
        texts: List[str],
        model: str = "ViT-B/32",
        device: str = "auto",
) -> dict:
    model, preprocess = ClipSingleton.get(model, device)
    device = model.logit_scale.device

    text_tokens = [
        clip.tokenize(text)
        for text in texts
    ]
    text_tokens = torch.cat(text_tokens).to(device)

    images = []
    for filename in filenames:
        image = load_image(filename)
        image = resize_crop(image, [224, 224])
        image = preprocess(image)
        images.append(image.unsqueeze(0))

    images = torch.cat(images)

    with torch.no_grad():
        similarities = model(images, text_tokens)[0]

    ret = dict()
    for filename, file_sim in zip(filenames, similarities):
        ret[filename] = dict()
        for text, sim in zip(texts, file_sim):
            ret[filename][text] = float(sim)

    return ret


def trim_length(s: str, max_len: int, front: bool = False) -> str:
    if len(s) > max_len:
        if front:
            return ".." + s[-max_len+2:]
        return s[:max_len-2] + ".."
    return s


def dump_similarities(similarities: dict, sort: Optional[int] = None):
    org_texts = list(similarities[next(iter(similarities.keys()))].keys())
    texts = [f"'{trim_length(t, 20)}'" for t in org_texts]

    max_filename_len = min(40, max(len(fn) for fn in similarities))
    max_text_len = max(len(t) for t in texts)

    filenames = list(similarities.keys())
    if sort:
        filenames.sort(key=lambda fn: similarities[fn][org_texts[abs(sort) - 1]], reverse=sort < 0)

    print(" " * (max_filename_len+1) + " ".join(f"{t:{max_text_len}}" for t in texts))
    for filename in filenames:
        sims = similarities[filename]
        print(f"{trim_length(filename, max_filename_len, True):{max_filename_len}} ", end="")
        sims = [f"{sim:.2f}" for sim in sims.values()]
        for sim in sims:
            print(f"{sim:{max_text_len}} ", end="")
        print()


if __name__ == "__main__":

    args = parse_arguments()

    similarities = rate_images(
        filenames=args.image,
        texts=args.text,
        model=args.model,
        device=args.device,
    )

    dump_similarities(
        similarities,
        sort=args.sort,
    )