import json
import re
import glob
from pathlib import Path
from typing import List, Optional

from rate import ClipRater
from scripts.helper import render


DEVICE = "auto"


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


RESO_ONE_PASS_SCRIPT = """
epochs: 300
resolution: 
  - '{0: 8, 1: 8, 2: 16, 3: 32, 4: 64, 5: 128}.get(int(t*16),
224)' 
optimizer: rmsprob
init:
  mean: 0.3
  std: .01
targets:
  - name: full scale
    batch_size: 10
    features:
      - text: {{text}}
    transforms:
      - resize: 224
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
          start: 50%
      - saturation:
          below: 0.005
"""

RESO_INIT_SCRIPT = """
epochs: 20
resolution: 
  - 8 if t < .4 else 16
optimizer: rmsprob
learnrate: .8
init:
  mean: 0.3
  std: .01
targets:
  - name: full scale
    batch_size: 10
    features:
      - text: {{text}}
    transforms:
      - resize: 224
      - pad:
          size: 32
          mode: edge
      - random_rotate:
          degree: -20 20
          center: 0 1
      - random_crop: 224
      - mul: 1/3
      - mean: 0.5
      - bwnoise: 1/5 
    constraints:
      - saturation:
          below: 0.005
"""

RESO_DETAIL_SCRIPT = """
epochs: 300
resolution: 
  - max(8,min(224, int(t*448/8)*8 ))  
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
      - resize: 224
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
          start: 45%
      - saturation:
          below: 0.005
"""


def get_prompts() -> List[dict]:
    with open(Path(__file__).resolve().parent / "dall-e-samples.json") as fp:
        data = json.load(fp)

    def _repl(m):
        return m.groups()[0]

    prompts = []
    for info in data["completion_info"]:
        text = info["text"]
        text = re.sub(r"{\d:([^}]+)}", _repl, text)
        keys = info["variable_assignments"]
        ascii_keys = [
            "".join(c if c.isalpha() else "-" for c in key)
            for key in keys
        ]
        prompts.append({
            "text": text,
            "keys": ascii_keys,
            "orig_keys": keys,
            "filename": f"{ascii_keys[0]}-of-{ascii_keys[1]}",
        })

    prompts.sort(key=lambda p: p["keys"])
    return prompts


def get_own_prompts() -> List[dict]:
    animals = [
        "macgyver",
        "h.p. lovecraft",
        "cthulhu",
        "bob dobbs",
        "james t. kirk",
        "a skull",
        "an evil teddy bear",
        "a friendly teddy bear",
    ]
    things = [
        "bits and bytes",
        "a bloody mess",
        "led zeppelin",
        "microbes",
        "furry dwarfs",
        "voxel graphics",
        "flowers in harmony",
        "philosophic contemplation",
        "fractal",
        "spiderweb",
    ]
    prompts = []
    for animal in sorted(animals):
        for thing in sorted(things):
            keys = [animal, thing]
            ascii_keys = [
                "".join(c if c.isalpha() else "-" for c in key)
                for key in keys
            ]
            prompts.append({
                "text": f"{animal} made of {thing}. {animal} with the texture of {thing}",
                "keys": ascii_keys,
                "orig_keys": keys,
                "filename": f"{ascii_keys[0]}-of-{ascii_keys[1]}",
            })
    return prompts


class Renderer:

    def __init__(
            self,
            out_dir: str,
            init_out_dir: Optional[str] = None,
            one_pass_script: Optional[str] = None,
            init_script: Optional[str] = None,
            detail_script: Optional[str] = None,
            prompts: Optional[List[dict]] = None
    ):
        self.prompts = prompts or get_prompts()
        self.output_path_base = (
            Path(__file__).resolve().parent.parent
            / "images" / "dalle"
        )
        self.output_path = self.output_path_base / out_dir
        self.init_output_path = None if init_out_dir is None else self.output_path_base / init_out_dir
        self.one_pass_script = one_pass_script
        self.init_script = init_script
        self.detail_script = detail_script

    def prompt(self, *keys: str) -> Optional[dict]:
        keys = list(keys)
        for p in self.prompts:
            if p["keys"] == keys:
                return p

    def render_one_pass(self, prompt: dict):
        output_name = self.output_path / f"{prompt['filename']}.png"

        if not output_name.exists():
            render(
                device=DEVICE,
                source=self.one_pass_script,
                output_name=output_name,
                template_context={"text": prompt["text"].split(".")[0] + "."}
            )

    def render_init(self, prompt: dict, count: int = 20):
        for i in range(count):
            output_name = (
                self.init_output_path / prompt['filename']
                / f"{prompt['filename']}-{i}.png"
            )
            if not output_name.exists():
                break
            count = count - 1

        if not output_name.exists():
            render(
                device=DEVICE,
                source=self.init_script,
                output_name=output_name,
                template_context={"text": prompt["text"]},
                extra_args=["--repeat", str(count)]
            )

    def render_detail(self, prompt: dict, init_filename: str, suffix: str = ""):
        output_name = (
            self.output_path / prompt['filename'] / f"{prompt['filename']}{suffix}.png"
        )
        if not output_name.exists():
            render(
                device=DEVICE,
                source=self.detail_script,
                output_name=output_name,
                template_context={
                    "text": prompt["text"],
                    "image": init_filename,
                },
                snapshot_interval=60.,
            )

    def render_all_one_pass(self):
        for p in self.prompts:
            self.render_one_pass(p)

    def render_all_init(self, count: int = 20):
        for p in self.prompts:
            self.render_init(p, count=count)

    def render_all_detail(self, num_best: int = 1, num_worst: int = 0, keys: Optional[List[str]] = None):
        for prompt in self.prompts:
            if keys and prompt["keys"] != keys:
                continue

            input_path = self.init_output_path / prompt['filename']
            filenames = sorted(glob.glob(str(input_path / "*.png")))
            if not filenames:
                continue

            rater = ClipRater(
                filenames=filenames,
                texts=[prompt["text"]],
                device="cpu",
                caching=True,
            )
            similarities = rater.rate()

            similarities = similarities.sort_values(by=similarities.columns[0], ascending=False)

            input_names = list(similarities.index[:num_best])
            if num_worst:
                input_names += list(similarities.index[-num_worst:])

            for i, input_name in enumerate(input_names):
                number = input_name[:-4].split("-")[-1]
                self.render_detail(
                    prompt,
                    init_filename=input_name,
                    suffix=f"-from-{number}",
                )

    def dump_prompts_json(self):
        things = dict()
        for p in self.prompts:
            animal, thing = p["orig_keys"]
            if thing not in things:
                things[thing] = dict()
            things[thing][animal] = p["filename"]

        print(json.dumps(things, indent=4))


def dump_dalle_image_urls():
    with open(Path(__file__).resolve().parent / "dall-e-samples.json") as fp:
        data = json.load(fp)
    urls = []
    for i in data["completion_info"]:
        if i["variable_assignments"] in (
            ["snail", "harp"],
            ["penguin", "piano"],
        ):
            url = i["completions"][0]["image_url_prefix"]
            for n in range(30):
                urls.append(f"{data['base_image_url']}/{url}_{n}.png")

    print("\n".join(urls))


if __name__ == "__main__":

    # dump_dalle_image_urls(); exit()

    if 0:
        renderer = Renderer(
            out_dir="own-onepass",
            one_pass_script=RESO_ONE_PASS_SCRIPT,
            prompts=get_own_prompts(),
        )
        renderer.render_all_one_pass()

    if 1:
        renderer = Renderer(
            out_dir="detail",
            init_out_dir="init",
            init_script=INIT_SCRIPT,
            detail_script=DETAIL_SCRIPT,
        )
        #renderer.render_init(renderer.prompt("penguin", "piano"), 512)
        #renderer.render_all_init(20)
        #renderer.render_all_detail(num_best=6, num_worst=2, keys=["penguin", "piano"])
        renderer.render_all_detail(num_best=6, num_worst=2, keys=["snail", "harp"])

    if 0:
        renderer = Renderer(
            out_dir="reso-detail",
            init_out_dir="reso-init",
            # init_script=INIT_SCRIPT,
            #init_script=RESO_INIT_SCRIPT,
            detail_script=RESO_DETAIL_SCRIPT,
        )
        #renderer.render_init(renderer.prompt("penguin", "piano"), 512)

        #renderer.render_all_init(20)
        renderer.render_all_detail(num_best=6, num_worst=2, keys=["penguin", "piano"])
