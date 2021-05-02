import sys
import math
import random
import traceback
import argparse
import time
from typing import Union, Sequence, Type, Tuple, Optional

import numpy as np
import torch
import torch.nn
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
from torchvision.utils import save_image, make_grid
import PIL.Image
import clip
from tqdm import tqdm

from .parameters import save_yaml_config
from .files import make_filename_dir, change_extension
from .pixel_models import PixelsRGB
from .expression import Expression, ExpressionContext
from . import transforms as transform_modules
from . import constraints as constraint_modules


torch.autograd.set_detect_anomaly(True)


class ImageTraining:

    clip_resolution = [224, 224]

    def __init__(self, parameters: dict):
        self.parameters = parameters

        if self.parameters["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.parameters["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Cuda device requested but not available")

        self._clip_model = None
        self.clip_preprocess = None

        self.pixel_model = PixelsRGB(parameters["resolution"]).to(self.device)

        self.learnrate = 0.01
        self.optimizer = torch.optim.Adam(
            self.pixel_model.parameters(),
            lr=self.learnrate,  # will be adjusted per epoch
        )

        self.targets = []

    @property
    def clip_model(self):
        if self._clip_model is None:
            self.log(2, "loading CLIP")
            self._clip_model, self.clip_preprocess = clip.load(
                self.parameters["model"],
                device=self.device
            )
        return self._clip_model

    @property
    def verbose(self):
        return self.parameters["verbose"]

    def log(self, level: int, *args):
        if self.verbose >= level:
            print(*args, file=sys.stderr)
            sys.stderr.flush()

    def save_image(self):
        filename = self.parameters["output"]
        self.log(2, f"saving {filename}")
        make_filename_dir(filename)
        save_image(self.pixel_model.forward(), filename)

    def setup_targets(self):
        self.log(2, "getting target features")
        self.targets = []
        for target_param in self.parameters["targets"]:
            if not target_param["active"]:
                continue

            # -- get target features ---

            features = []
            weights = []
            for feature_param in target_param["features"]:
                if feature_param.get("text"):
                    tokens = clip.tokenize([feature_param["text"]]).to(self.device)
                    feature = self.clip_model.encode_text(tokens)
                else:
                    image = PIL.Image.open(feature_param["image"])
                    image = self.clip_preprocess(image)
                    feature = self.clip_model.encode_image(image.unsqueeze(0))

                features.append(feature)
                weights.append(feature_param["weight"])

            if features:
                features = torch.cat(features)
                features = features / features.norm(dim=-1, keepdim=True)

            target = {
                "params": target_param,
                "features": features,
                "weights": torch.Tensor(weights).to(self.device) if weights else [],
                "loss_function": torch.nn.MSELoss(),

                "feature_losses": [ValueQueue() for _ in features],
                "feature_similarities": [ValueQueue() for _ in features],
            }

            # --- setup transforms ---

            transforms = []
            final_resolution = self.parameters["resolution"].copy()
            for trans_param in target_param["transforms"]:
                if trans_param.get("repeat"):
                    transforms.append(
                        transform_modules.RepeatTransform(trans_param["repeat"])
                    )

                if trans_param.get("blur"):
                    p = trans_param["blur"]
                    transforms.append(VT.GaussianBlur(int(p[0]), [p[1], p[1]]))

                affine_kwargs = dict()
                if trans_param.get("random_translate"):
                    affine_kwargs["translate"] = trans_param["random_translate"]
                if trans_param.get("random_scale"):
                    affine_kwargs["scale"] = trans_param["random_scale"]

                if affine_kwargs:
                    affine_kwargs["degrees"] = 0
                    affine_kwargs["fillcolor"] = None
                    transforms.append(VT.RandomAffine(**affine_kwargs))

                if trans_param.get("random_rotate"):
                    transforms.append(
                        VT.RandomRotation(
                            degrees=trans_param["random_rotate"]["degree"],
                            center=trans_param["random_rotate"]["center"],
                        )
                    )

                if trans_param.get("random_crop"):
                    transforms.append(VT.RandomCrop(trans_param["random_crop"]))
                    final_resolution = trans_param["random_crop"]

                if trans_param.get("resize"):
                    transforms.append(VT.Resize(trans_param["resize"]))
                    final_resolution = trans_param["resize"]

                if trans_param.get("center_crop"):
                    transforms.append(VT.CenterCrop(trans_param["center_crop"]))
                    final_resolution = trans_param["center_crop"]

                if trans_param.get("noise"):
                    transforms.append(transform_modules.NoiseTransform(trans_param["noise"]))

            if final_resolution != self.clip_resolution:
                transforms.append(VT.Resize(self.clip_resolution))

            if transforms:
                target["transforms"] = torch.nn.Sequential(*transforms).to(self.device)
                self.log(2, f"target '{target_param['name']}' transforms:")
                self.log(2, target["transforms"])

            # --- setup constraints ---

            target["constraints"] = []
            for constr_param in target_param["constraints"]:
                constraint = None
                if constr_param.get("mean"):
                    constraint = constraint_modules.MeanConstraint(**constr_param["mean"])
                if constr_param.get("std"):
                    constraint = constraint_modules.StdConstraint(**constr_param["std"])

                if constraint:
                    target["constraints"].append({
                        "params": constr_param,
                        "model": constraint.to(self.device),
                        "losses": ValueQueue(),
                    })

            self.targets.append(target)

    def initialize(self):
        self.log(2, "initializing pixels")
        self.pixel_model.initialize(self.parameters["init"])

    def train(self):
        assert self.clip_model

        if not self.targets:
            self.setup_targets()

        self.initialize()

        last_stats_time = 0
        last_snapshot_time = 0
        loss_queue = ValueQueue()

        epoch_iter = range(self.parameters["epochs"])
        if self.verbose >= 1:
            epoch_iter = tqdm(epoch_iter)

        self.log(2, "training")
        for epoch in epoch_iter:
            epoch_f = epoch / max(1, self.parameters["epochs"] - 1)

            expression_ctx = ExpressionContext(epoch=epoch, t=epoch_f)

            # --- update learnrate ---

            learnrate = self.learnrate * expression_ctx(self.parameters["learnrate"])

            for g in self.optimizer.param_groups:
                g['lr'] = learnrate

            # --- post process pixels ---

            self._postproc(epoch, epoch_f)

            # --- get pixels ---

            current_pixels = self.pixel_model.forward()

            # --- combine loss for each target ---

            loss = torch.tensor(0).to(self.device)

            for target in self.targets:
                if not _check_start_end(
                    target["params"]["start"], target["params"]["end"],
                    epoch, epoch_f
                ):
                    continue

                pixels = current_pixels
                if target.get("transforms"):
                    pixels = target["transforms"](pixels)

                norm_pixels = self.clip_preprocess.transforms[-1](pixels.unsqueeze(0))
                clip_features = self.clip_model.encode_image(norm_pixels)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

                loss = loss + (
                    target["params"]["weight"] * self._get_target_loss(
                        target, pixels, clip_features
                    )
                )

            # --- adjust weights ---

            if loss:
                loss_queue.append(float(loss))
                self.pixel_model.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

            # --- print stats ---

            cur_time = time.time()

            if self.verbose >= 2 and cur_time - last_stats_time > 4:
                last_stats_time = cur_time
                mean = [round(float(f), 3) for f in current_pixels.reshape(3, -1).mean(1)]
                std = [round(float(f), 3) for f in current_pixels.reshape(3, -1).std(1)]
                self.log(2, f"--- train step {epoch+1} / {self.parameters['epochs']} ---")
                self.log(2, f"device {self.device}")
                self.log(2, f"resolution {self.parameters['resolution']}, mean {mean}, std {std}")
                self.log(2, f"learnrate {learnrate:.3f}, loss {loss_queue}")
                self.log(2, "targets:")
                self.print_target_stats()

            if epoch == 0 or cur_time - last_snapshot_time > 10:
                last_snapshot_time = cur_time
                self.save_image()

    def _postproc(self, epoch: int, epoch_f: float):
        for pp in self.parameters["postproc"]:
            if not pp["active"]:
                continue

            if not _check_start_end(pp["start"], pp["end"], epoch, epoch_f):
                continue

            if pp.get("blur"):
                self.pixel_model.blur(int(pp["blur"][0]), pp["blur"][1])

    def _get_target_loss(
            self,
            target: dict,
            pixels: torch.Tensor,
            clip_features: torch.Tensor,
    ) -> torch.Tensor:
        loss_sum = torch.tensor(0).to(self.device).to(clip_features.dtype)

        #color_diff = (
        #    torch.Tensor([1, 1, 0]).to(self.device)
        #    - pixels.reshape(3, -1).mean(-1)
        #)
        #loss_sum += 0.05 * color_diff @ color_diff.T

        if isinstance(target["features"], torch.Tensor):
            target_features: torch.Tensor = target["features"]

            similarities = 100. * target_features @ clip_features.T

            for i, target_feature in enumerate(target_features):
                loss = 100. * target["loss_function"](clip_features[0], target_feature)

                loss_sum += target["weights"][i] * loss

                # track statistics
                target["feature_losses"][i].append(float(loss))
                target["feature_similarities"][i].append(float(similarities[i]))

        for constraint in target.get("constraints", []):
            loss = constraint["model"].forward(pixels)
            constraint["losses"].append(float(loss))
            loss_sum += loss

        return loss_sum

    def print_target_stats(self):
        feature_length = 40

        rows = []
        for target in self.targets:
            for i, f in enumerate(target["params"]["features"]):
                row = {
                    "name": _short_str(target["params"]["name"], 30) if i == 0 else "",
                    "weight": round(target["params"]["weight"], 3),
                }
                if f.get("text"):
                    row["feature"] = _short_str(f["text"], feature_length)
                else:
                    row["feature"] = _short_str(f["image"], feature_length, True)

                for name, queue in (
                        ("loss", target["feature_losses"][i]),
                        ("sim", target["feature_similarities"][i]),
                ):
                    row[f"{name}_mean"] = round(queue.mean(), 3)
                    row[f"{name}_min"] = round(queue.min(), 3)
                    row[f"{name}_max"] = round(queue.max(), 3)

                rows.append(row)

            for i, constraint in enumerate(target["constraints"]):
                row = {
                    "name": "constraint" if i == 0 else "",
                    "weight": round(constraint["model"].weight, 3),
                    "feature": _short_str(str(constraint["model"]), feature_length),
                    "loss_mean": round(constraint["losses"].mean(), 3),
                    "loss_min": round(constraint["losses"].min(), 3),
                    "loss_max": round(constraint["losses"].max(), 3),
                }
                rows.append(row)

        rows = [{k: str(v) for k, v in row.items()} for row in rows]
        all_keys = set(sum((list(row.keys()) for row in rows), []))
        lengths = {
            key: max(len(r.get(key) or "") for r in rows)
            for key in all_keys
        }
        for row in rows:
            line = (
                f"""{row["name"]:{lengths["name"]}} : """
                f"""{row["weight"]:{lengths["weight"]}} x """
                f"""{row["feature"]:{lengths["feature"]}} : """
                f"""loss {row["loss_mean"]:{lengths["loss_mean"]}} ("""
                f"""{row["loss_min"]:{lengths["loss_min"]}} - """
                f"""{row["loss_max"]:{lengths["loss_max"]}})"""
            )
            if row.get("sim_mean"):
                line += (
                    f""" / sim {row["sim_mean"]:{lengths["sim_mean"]}} ("""
                    f"""{row["sim_min"]:{lengths["sim_min"]}} - """
                    f"""{row["sim_max"]:{lengths["sim_max"]}})"""
                )
            self.log(0, line)


def _check_start_end(
        start: Union[int, float],
        end: Union[int, float],
        epoch: int,
        epoch_f: float,
):
    if isinstance(start, int) and start > epoch:
        return False
    elif isinstance(start, float) and start > epoch_f:
        return False
    if isinstance(end, int) and end < epoch:
        return False
    elif isinstance(end, float) and end < epoch_f:
        return False

    return True


def _short_str(s: str, max_length: int, front: bool = False) -> str:
    assert max_length > 2
    if len(s) <= max_length:
        return s

    if front:
        return ".." + s[-max_length + 2:]
    else:
        return s[:max_length - 2] + ".."


class ValueQueue:
    def __init__(self, max_length: int = 10):
        self.max_length = max_length
        self.values = list()

    def __str__(self):
        return f"{self.mean():.3f} ({self.min():.3f} - {self.max():.3f})"

    def append(self, v):
        self.values.append(v)
        if len(self.values) > self.max_length:
            self.values.pop(0)

    def mean(self):
        return 0. if not self.values else sum(self.values) / len(self.values)

    def min(self):
        return 0. if not self.values else min(self.values)

    def max(self):
        return 0. if not self.values else max(self.values)

