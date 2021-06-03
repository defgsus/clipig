import json
import re
import glob
from pathlib import Path
from typing import List

from rate import ClipRater
from scripts.helper import render

INIT_SCRIPT = """
epochs: 20
optimizer: rmsprob
learnrate: 2
init:
  mean: 0.33
  std: 0.03
  resolution: 10
targets:
  - name: full scale
    batch_size: 10
    features:
      - text: {{text}}
    transforms:
      - pad:
          size: 32
          mode: edge
      - random_rotate:
          degree: -30 30
          center: 0 1
      - mul: 1/2
      - mean: 0.5
      - noise: 1/5
    constraints:
      - blur:
          kernel_size: 11
          weight: .4
"""


DETAIL_SCRIPT = """
epochs: 300
optimizer: rmsprob
init:
  image: {{image}}
  mean: 0
  std: 1
targets:
  - name: full scale
    batch_size: 10
    features:
      - text: {{text}}
    transforms:
      - pad:
          size: 32
          mode: edge
      - random_rotate:
          degree: -20 20
          center: 0 1
      - random_crop: 224
      - mul: 1/3
      - mean: 0.5
      - bwnoise: 1/5*ti
    constraints:
      - blur: 
          kernel_size: 11
          start: 70%
      - saturation:
          below: 0.005
"""


def get_prompts():
    with open(Path(__file__).resolve().parent / "dall-e-samples.json") as fp:
        data = json.load(fp)

    def _repl(m):
        return m.groups()[0]

    prompts = []
    for info in data["completion_info"]:
        text = info["text"]
        text = re.sub(r"{\d:([^}]+)}", _repl, text)
        keys = info["variable_assignments"]
        keys = [
            "".join(c if c.isalpha() else "-" for c in key)
            for key in keys
        ]
        prompts.append({
            "text": text,
            "keys": keys,
            "filename": f"{keys[0]}-of-{keys[1]}",
        })

    return prompts


def render_single_init(prompt: dict, count: int = 20):
    for i in range(count):
        output_name = (
                Path(__file__).resolve().parent.parent
                / "images" / "dalle" / "init" / prompt['filename']
                / f"{prompt['filename']}-{i}.png"
        )
        if not output_name.exists():
            render(
                source=INIT_SCRIPT,
                output_name=output_name,
                template_context={"text": prompt["text"]}
            )


def render_single_detail(prompt: dict, init_filename: str):
    output_name = (
            Path(__file__).resolve().parent.parent
            / "images" / "dalle" / "dall-e-samples"
            / f"{prompt['filename']}.png"
    )
    if not output_name.exists():
        render(
            source=DETAIL_SCRIPT,
            output_name=output_name,
            template_context={
                "text": prompt["text"],
                "image": init_filename,
            },
        )


def render_all_init(prompts: List[dict], count: int = 20):
    for p in prompts:
        render_single_init(p, count=count)


def render_all_detail(prompts: List[dict], count: int = 20):
    for prompt in prompts:
        input_path = (
            Path(__file__).resolve().parent.parent
            / "images" / "dalle" / "init" / prompt['filename']
        )
        filenames = sorted(glob.glob(str(input_path / "*.png")))
        if not filenames:
            break

        rater = ClipRater(
            filenames=filenames,
            texts=[prompt["text"]],
        )
        similarities = rater.rate()

        similarities = similarities.sort_values(by=similarities.columns[0])

        input_name = similarities.index[0]

        render_single_detail(prompt, init_filename=input_name)


if __name__ == "__main__":

    prompts = get_prompts()

    # render_all_init(prompts, 20)
    render_all_detail(prompts)
