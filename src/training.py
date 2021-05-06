import sys
import math
import random
import traceback
import argparse
import time
from typing import Union, Sequence, Type, Tuple, Optional, Callable, List

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
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
from .constraints import get_mean_saturation
from .clip_singleton import ClipSingleton


torch.autograd.set_detect_anomaly(True)


class ImageTraining:

    clip_resolution = [224, 224]

    def __init__(
            self,
            parameters: dict,
            snapshot_callback: Optional[Callable] = None,
            log_callback: Optional[Callable] = None,
            progress_callback: Optional[Callable] = None,
    ):
        self.parameters = parameters
        self.snapshot_callback = snapshot_callback
        self.log_callback = log_callback
        self.progress_callback = progress_callback

        if self.parameters["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.parameters["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Cuda device requested but not available")

        self._clip_model = None
        self.clip_preprocess = None

        self.pixel_model = PixelsRGB(parameters["resolution"]).to(self.device)

        self.epoch = self.parameters["start_epoch"]

        # for threaded access
        self._stop = False
        self._running = False

        if self.parameters["optimizer"] == "adam":
            self.base_learnrate = 0.01
            self.optimizer = torch.optim.Adam(
                self.pixel_model.parameters(),
                lr=self.base_learnrate,  # will be adjusted per epoch
            )
        elif self.parameters["optimizer"] == "sgd":
            self.base_learnrate = 50.0
            self.optimizer = torch.optim.SGD(
                self.pixel_model.parameters(),
                lr=self.base_learnrate,  # will be adjusted per epoch
            )
        elif self.parameters["optimizer"] == "sparse_adam":
            self.base_learnrate = 0.01
            self.optimizer = torch.optim.RMSprop(
                self.pixel_model.parameters(),
                lr=self.base_learnrate,  # will be adjusted per epoch
            )
        elif self.parameters["optimizer"] == "adadelta":
            self.base_learnrate = 50.0
            self.optimizer = torch.optim.Adadelta(
                self.pixel_model.parameters(),
                lr=self.base_learnrate,  # will be adjusted per epoch
            )
        elif self.parameters["optimizer"] == "rmsprob":
            self.base_learnrate = 0.01
            self.optimizer = torch.optim.RMSprop(
                self.pixel_model.parameters(),
                lr=self.base_learnrate,  # will be adjusted per epoch
                centered=True,
                # TODO: high momentum is quite useful for more 'chaotic' images but needs to
                #   be adjustable by config expression
                momentum=0.1,
            )
        else:
            raise ValueError(f"Unknown optimizer '{self.parameters['optimizer']}'")

        self.targets = []

    @property
    def clip_model(self):
        if self._clip_model is None:
            self.log(2, "loading CLIP")
            self._clip_model, self.clip_preprocess = ClipSingleton.get(
                model=self.parameters["model"],
                device=self.device
            )
        return self._clip_model

    @property
    def verbose(self):
        return self.parameters["verbose"]

    def log(self, level: int, *args):
        if self.verbose >= level:
            if self.log_callback is not None:
                self.log_callback(*args)
            else:
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
            feature_loss_functions = []
            for feature_param in target_param["features"]:
                if feature_param.get("text"):
                    tokens = clip.tokenize([feature_param["text"]]).to(self.device)
                    feature = self.clip_model.encode_text(tokens)
                else:
                    image = PIL.Image.open(feature_param["image"])
                    image = self.clip_preprocess(image)
                    feature = self.clip_model.encode_image(image.unsqueeze(0))

                features.append(feature)

                feature_loss_functions.append(get_feature_loss_function(feature_param["loss"]))

            if features:
                features = torch.cat(features)
                features = features / features.norm(dim=-1, keepdim=True)

            target = {
                "params": target_param,
                "features": features,
                "feature_loss_functions": feature_loss_functions,

                # statistics ...
                "count": 0,
                "feature_losses": [ValueQueue() for _ in features],
                "feature_similarities": [ValueQueue() for _ in features],
            }

            # --- setup transforms ---

            is_random = False  # determine if the transform stack includes randomization
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
                    is_random = True
                    affine_kwargs["translate"] = trans_param["random_translate"]

                if trans_param.get("random_scale"):
                    is_random = True
                    affine_kwargs["scale"] = trans_param["random_scale"]

                if affine_kwargs:
                    affine_kwargs["degrees"] = 0
                    affine_kwargs["fillcolor"] = None
                    transforms.append(VT.RandomAffine(**affine_kwargs))

                if trans_param.get("random_rotate"):
                    is_random = True
                    transforms.append(
                        VT.RandomRotation(
                            degrees=trans_param["random_rotate"]["degree"],
                            center=trans_param["random_rotate"]["center"],
                        )
                    )

                if trans_param.get("random_crop"):
                    is_random = True
                    transforms.append(VT.RandomCrop(trans_param["random_crop"]))
                    final_resolution = trans_param["random_crop"]

                if trans_param.get("resize"):
                    transforms.append(VT.Resize(trans_param["resize"]))
                    final_resolution = trans_param["resize"]

                if trans_param.get("center_crop"):
                    transforms.append(VT.CenterCrop(trans_param["center_crop"]))
                    final_resolution = trans_param["center_crop"]

                if trans_param.get("noise"):
                    is_random = True
                    transforms.append(transform_modules.NoiseTransform(trans_param["noise"]))

                if trans_param.get("edge"):
                    transforms.append(transform_modules.EdgeTransform(trans_param["edge"]))

            if final_resolution != self.clip_resolution:
                transforms.append(VT.Resize(self.clip_resolution))

            if transforms:
                target["is_random"] = is_random
                target["transforms"] = torch.nn.Sequential(*transforms).to(self.device)
                self.log(2, f"target '{target_param['name']}' transforms:")
                self.log(2, target["transforms"])

            # --- setup constraints ---

            target["constraints"] = []
            for constr_param in target_param["constraints"]:
                for name, Module in (
                        ("mean", constraint_modules.MeanConstraint),
                        ("std", constraint_modules.StdConstraint),
                        ("edge_max", constraint_modules.EdgeMaxConstraint),
                        ("saturation", constraint_modules.SaturationConstraint),
                        ("blur", constraint_modules.BlurConstraint),
                ):
                    if constr_param.get(name):
                        constraint = Module(**constr_param[name])

                        target["constraints"].append({
                            "params": constr_param,
                            "model": constraint.to(self.device),

                            # statistics ...
                            "losses": ValueQueue(),
                        })

            self.targets.append(target)

    def initialize(self):
        self.log(2, "initializing pixels")
        self.pixel_model.initialize(self.parameters["init"])

    def train(self, initialize: bool = True):
        self._running = True
        try:
            self._train(initialize=initialize)
        except:
            self._running = False
            raise
        self._running = False

    def _train(self, initialize: bool = True):
        assert self.clip_model

        if not self.targets:
            self.setup_targets()

        if initialize:
            self.initialize()

        last_stats_time = 0
        last_snapshot_time = 0
        last_snapshot_epoch = 0
        loss_queue = ValueQueue()

        epoch_iter = range(self.epoch, self.parameters["epochs"])
        if self.verbose >= 1 and self.log_callback is None:
            epoch_iter = tqdm(epoch_iter)

        self.log(2, "training")
        for epoch in epoch_iter:

            if self._stop:
                break

            self.epoch = epoch
            epoch_f = epoch / max(1, self.parameters["epochs"] - 1)

            expression_context = ExpressionContext(epoch=epoch, epoch_f=epoch_f, t=epoch_f)

            # --- update learnrate ---

            learnrate_scale = expression_context(self.parameters["learnrate_scale"])
            learnrate = self.base_learnrate * learnrate_scale * expression_context(self.parameters["learnrate"])

            for g in self.optimizer.param_groups:
                g['lr'] = learnrate

            expression_context = expression_context.add(
                lr=learnrate, learnrate=learnrate,
                lrs=learnrate_scale, learnrate_scale=learnrate_scale,
            )

            # --- post process pixels ---

            self._postproc(epoch, epoch_f, expression_context)

            # --- get pixels ---

            current_pixels = self.pixel_model.forward()

            # --- apply target transforms ---

            active_targets = []
            target_pixels = []
            target_to_pixels_mapping = dict()
            for target in self.targets:
                if _check_start_end(
                        target["params"]["start"], target["params"]["end"],
                        epoch, epoch_f
                ):
                    active_targets.append(target)
                    target_idx = len(active_targets) - 1
                    for batch_idx in range(target["params"]["batch_size"]):
                        if not target["is_random"] and target_idx in target_to_pixels_mapping:
                            continue

                        pixels = current_pixels
                        if target.get("transforms"):
                            pixels = target["transforms"](pixels)

                        target_pixels.append(pixels.unsqueeze(0))
                        target_pixel_idx = len(target_pixels) - 1

                        if target_idx not in target_to_pixels_mapping:
                            target_to_pixels_mapping[target_idx] = [target_pixel_idx]
                        else:
                            target_to_pixels_mapping[target_idx].append(target_pixel_idx)

            if active_targets:
                target_pixels = torch.cat(target_pixels, dim=0)

                # --- retrieve CLIP features ---

                norm_pixels = self.clip_preprocess.transforms[-1](target_pixels)
                clip_features = self.clip_model.encode_image(norm_pixels)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)

                # --- combine loss for each target ---

                loss = torch.tensor(0).to(self.device)

                for target_idx, target_pixel_indices in target_to_pixels_mapping.items():
                    target = active_targets[target_idx]
                    for target_pixel_idx in target_pixel_indices:
                        pixels = target_pixels[target_pixel_idx]
                        clip_feature = clip_features[target_pixel_idx]

                        loss = loss + (
                            self._get_target_loss(
                                target, pixels, clip_feature, expression_context
                            )
                        )

                # --- adjust weights ---

                if loss:
                    loss_queue.append(float(loss))
                    self.pixel_model.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

            # --- post stats ---

            if self.progress_callback is not None:
                self._post_epoch_statistics(
                    current_pixels=current_pixels,
                    learnrate=learnrate,
                    learnrate_scale=learnrate_scale,
                    context=expression_context,
                    loss_queue=loss_queue,
                )

            # --- print stats ---

            cur_time = time.time()

            if self.verbose >= 2 and (cur_time - last_stats_time > 4) or epoch == self.parameters["epochs"] - 1:
                last_stats_time = cur_time
                mean = [round(float(f), 3) for f in current_pixels.reshape(3, -1).mean(1)]
                std = [round(float(f), 3) for f in current_pixels.reshape(3, -1).std(1)]
                sat = round(float(get_mean_saturation(current_pixels)), 3)
                edge_max = [round(float(f), 3) for f in constraint_modules.get_edge_max(current_pixels)]
                self.log(2, f"--- train step {epoch+1} / {self.parameters['epochs']} ---")
                self.log(2, f"device: {self.device}")
                self.log(2, f"image: res {self.parameters['resolution']}, "
                            f"mean {mean}, std {std}, sat {sat}, edge_max {edge_max}")
                self.log(2, f"learning: optim '{self.parameters['optimizer']}', "
                            f"learnrate {learnrate:.3f} (scale {learnrate_scale:.3f}), "
                            f"loss {loss_queue}")
                self.log(2, "targets:")
                self._print_target_stats(expression_context)

            # -- store snapshot --

            do_snapshot = epoch == 0 or epoch == self.parameters["epochs"] - 1

            if isinstance(self.parameters["snapshot_interval"], int):
                do_snapshot |= epoch - last_snapshot_epoch >= self.parameters["snapshot_interval"]
            elif isinstance(self.parameters["snapshot_interval"], float):
                do_snapshot |= cur_time - last_snapshot_time >= self.parameters["snapshot_interval"]

            if epoch == 0 or do_snapshot:
                last_snapshot_time = cur_time
                last_snapshot_epoch = epoch
                if self.snapshot_callback is None:
                    self.save_image()
                else:
                    self.snapshot_callback(current_pixels)

    def _postproc(self, epoch: int, epoch_f: float, context: ExpressionContext):
        for pp in self.parameters["postproc"]:
            if not pp["active"]:
                continue

            if not _check_start_end(pp["start"], pp["end"], epoch, epoch_f):
                continue

            if pp.get("blur"):
                self.pixel_model.blur(
                    int(context(pp["blur"][0])),
                    context(pp["blur"][1]),
                )

            if pp.get("add"):
                self.pixel_model.add(context(pp["add"]))

            if pp.get("multiply"):
                self.pixel_model.multiply(context(pp["multiply"]))

    def _get_target_loss(
            self,
            target: dict,
            pixels: torch.Tensor,
            clip_feature: torch.Tensor,
            context: ExpressionContext,
    ) -> torch.Tensor:
        target["count"] += 1

        loss_sum = torch.tensor(0).to(self.device).to(clip_feature.dtype)

        if isinstance(target["features"], torch.Tensor):
            target_features: torch.Tensor = target["features"]

            similarities = 100. * target_features @ clip_feature.T

            feature_weights = self._get_target_feature_weights(target, similarities, context)

            target["applied_feature_weights"] = []
            for i, target_feature in enumerate(target_features):
                feature_weight, apply_feature = feature_weights[i]

                if not apply_feature:
                    target["applied_feature_weights"].append(feature_weight)
                    target["feature_losses"][i].append(0, count=False)
                    target["feature_similarities"][i].append(float(similarities[i]), count=False)

                else:
                    loss_function = target["feature_loss_functions"][i]

                    loss = loss_function(clip_feature, target_feature)

                    loss_sum += feature_weight * loss

                    # track statistics
                    target["applied_feature_weights"].append(feature_weight)
                    target["feature_losses"][i].append(float(loss))
                    target["feature_similarities"][i].append(float(similarities[i]))

            mean_sim = float(similarities.mean())
            context = context.add(sim=mean_sim, similarity=mean_sim)
        else:
            context = context.add(sim=0., similarity=0.)

        for constraint in target.get("constraints", []):
            loss = constraint["model"].forward(pixels, context)
            constraint["losses"].append(float(loss))
            loss_sum += loss

        return loss_sum * context(target["params"]["weight"])

    def _get_target_feature_weights(
            self,
            target: dict,
            similarities: torch.Tensor,
            context: ExpressionContext
    ) -> List[Tuple[float, bool]]:
        mode = target["params"]["select"]
        feature_weights = []

        if mode == "all":
            for i, target_feature in enumerate(target["features"]):
                feature_context = context.add(sim=float(similarities[i]), similarity=float(similarities[i]))
                weight = feature_context(target["params"]["features"][i]["weight"])
                feature_weights.append((weight, True))

        elif mode == "best":
            best_index = int(torch.argmax(similarities))
            for i, target_feature in enumerate(target["features"]):
                feature_context = context.add(sim=float(similarities[i]), similarity=float(similarities[i]))
                weight = feature_context(target["params"]["features"][i]["weight"])
                feature_weights.append((weight, i == best_index))

        else:
            raise ValueError(f"Unknown feature selection mode '{mode}'")

        return feature_weights

    def _post_epoch_statistics(
            self,
            current_pixels: torch.Tensor,
            learnrate: float,
            learnrate_scale: float,
            loss_queue: "ValueQueue",
            context: ExpressionContext,
    ):
        if self.progress_callback is None:
            return

        stats = {
            "epochs": self.parameters["epochs"],
            "epoch": self.epoch,
            "percent": self.epoch / max(1, self.parameters["epochs"] - 1) * 100.,
            #"image_mean": current_pixels.reshape(3, -1).mean(1).tolist(),
            #"image_std": current_pixels.reshape(3, -1).std(1).tolist(),
            #"image_sat": float(get_mean_saturation(current_pixels)),
            #"image_edge_max": constraint_modules.get_edge_max(current_pixels),
            "learnrate": learnrate,
            "learnrate_scale": learnrate_scale,
            "loss": loss_queue.last(),
        }
        self.progress_callback(stats)

    def _print_target_stats(self, context: ExpressionContext):
        feature_length = 40

        rows = []
        for target in self.targets:
            if target.get("applied_feature_weights"):
                for i, f in enumerate(target["params"]["features"]):
                    row = {
                        "name": _short_str(target["params"]["name"], 30) if i == 0 else "",
                        "weight": round(target["applied_feature_weights"][i], 3),
                    }
                    if f.get("text"):
                        row["feature"] = _short_str(f["text"], feature_length)
                    else:
                        row["feature"] = _short_str(f["image"], feature_length, True)

                    row["count"] = target["feature_losses"][i].count
                    row["count_p"] = round(target["feature_losses"][i].count / max(1, target["count"]) * 100., 1)
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
                    "name": "  constraint",
                    "weight": round(context(constraint["model"].weight), 3),
                    "feature": _short_str(constraint["model"].description(context), feature_length),
                    "count": constraint["losses"].count,
                    "count_p": round(constraint["losses"].count / max(1, target["count"]) * 100., 1),
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
                f"""count {row["count"]:{lengths["count"]}} """
                f"""({row["count_p"]:{lengths["count_p"]}}%) / """
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
    """Just some little statistics helper"""
    def __init__(self, max_length: int = 30):
        self.max_length = max_length
        self.values = list()
        self.count = 0

    def __str__(self):
        return f"{self.mean():.3f} ({self.min():.3f} - {self.max():.3f})"

    def append(self, v, count: bool = True):
        self.values.append(v)
        if count:
            self.count += 1
        if len(self.values) > self.max_length:
            self.values.pop(0)

    def mean(self):
        values = self.values[-self.max_length:]
        return 0. if not values else sum(values) / len(values)

    def min(self):
        values = self.values[-self.max_length:]
        return 0. if not values else min(values)

    def max(self):
        values = self.values[-self.max_length:]
        return 0. if not values else max(values)

    def last(self):
        return 0. if not self.values else self.values[-1]


def get_feature_loss_function(name: str) -> Callable:
    name = name.lower()

    if name in ("l1", "mae"):
        return lambda x1, x2: F.l1_loss(x1, x2) * 100.

    elif name in ("l2", "mse"):
        return lambda x1, x2: F.mse_loss(x1, x2) * 100.

    elif name in ("cosine", ):
        return lambda x1, x2: 1. - F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0))[0]

    else:
        raise ValueError(f"Invalid loss function '{name}'")

