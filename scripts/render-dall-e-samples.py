import json
import re
from pathlib import Path
from typing import List

from scripts.helper import render


CONFIG = """
epochs: 200
optimizer: rmsprob
init:
  mean: 0.33
  std: 0.03
  resolution: 10
targets:
  - name: full scale
    batch_size: 10
    select: best
    #feature_scale: same
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
      - noise: 1/10
postproc:
  - blur:
      kernel_size: 3
    end: .1
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


def render_prompts(prompts: List[dict]):
    for prompt in prompts:
        text = prompt["text"]
        output_name = (
                Path(__file__).resolve().parent.parent
                / "images" / "dall-e-samples"
                / f"{prompt['filename']}.png"
        )
        if not output_name.exists():
            render(
                source=CONFIG,
                output_name=output_name,
                template_context={"text": text}
            )


if __name__ == "__main__":

    prompts = get_prompts()
    render_prompts(prompts)
