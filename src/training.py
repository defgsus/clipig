import sys
import math
import random
import traceback
import argparse
import datetime
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
from .optimizers import create_optimizer
from .expression import Expression, ExpressionContext
from . import transforms as transform_modules
from . import constraints as constraint_modules
from .constraints import get_mean_saturation
from .clip_singleton import ClipSingleton
from .strings import value_str
from .images import load_image


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
        self.average_frame_rate = 0.
        self.training_seconds = 0.
        self.forward_seconds = 0.
        self.backward_seconds = 0.
        self.last_target_stats = ""

        if self.parameters["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.parameters["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Cuda device requested but not available")

        self._clip_model = None
        self.clip_preprocess = None

        self.pixel_model: PixelsRGB = None
        self.optimizer = None
        self.base_learnrate = None

        self.epoch = self.parameters["start_epoch"]
        self.epoch_f = self.epoch / max(1, self.parameters["epochs"] - 1)

        # for threaded access
        self._stop = False
        self._running = False

        self.targets = []
        self.postprocs = []

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

    def save_yaml(
            self,
            filename: str,
            run_time: float,
            epoch: Optional[int] = None
    ):
        if epoch is None:
            epoch = self.epoch

        self.log(2, f"exporting config {filename}")
        make_filename_dir(filename)
        save_yaml_config(
            filename, self.parameters,
            header=self.get_config_header(run_time=run_time, epoch=epoch),
        )

    def get_config_header(self, run_time: float, epoch: int,):
        from .doc import strip_doc

        epoch_time = run_time / max(1, self.epoch)

        header = strip_doc(f"""
        auto-generated at {datetime.datetime.now().replace(microsecond=0)}
        epochs: {epoch} / {self.parameters['epochs']}
        runtime: {run_time:.2f} seconds ({epoch_time:.3f}/epoch)
        """)
        if self.last_target_stats:
            header += f"\ntargets:\n{self.last_target_stats}"

        return "\n".join(
            f"# {line}"
            for line in header.splitlines()
        )

    def setup_targets(self):
        self.log(2, "getting target features")
        self.targets = []

        # --- setup post-processing transforms ---

        self.postprocs = []

        for param in self.parameters["postproc"]:
            for name, klass in transform_modules.transformations.items():
                if not param.get(name):
                    continue

                assert not klass.IS_RESIZE, f"{klass} changes resolution and can not be used for post-processing"
                transform_param = param[name]

                if isinstance(transform_param, dict):
                    t = klass(**transform_param)
                else:
                    t = klass(transform_param)

                self.postprocs.append({
                    "params": param,
                    "transform": t,
                })

        # --- targets ---

        for target_param in self.parameters["targets"]:
            if not target_param["active"]:
                continue

            # -- get target features ---

            features = []
            feature_loss_functions = []
            feature_start_ends = []
            feature_auto_scales = None
            feature_user_scales = []
            for feature_param in target_param["features"]:
                if feature_param.get("text"):
                    tokens = clip.tokenize([feature_param["text"]]).to(self.device)
                    feature = self.clip_model.encode_text(tokens)
                else:
                    image = load_image(feature_param["image"])
                    image = self.clip_preprocess(image)
                    feature = self.clip_model.encode_image(image.unsqueeze(0))

                features.append(feature)
                feature_loss_functions.append(get_feature_loss_function(feature_param["loss"]))
                feature_start_ends.append((
                    feature_param.get("start") or 0.,
                    feature_param.get("end") or 0.,
                ))
                feature_user_scales.append(feature_param["scale"])

            if features:
                features = torch.cat(features)
                features = features / features.norm(dim=-1, keepdim=True)
                if target_param["feature_scale"] != "equal":
                    feature_auto_scales = self._get_features_scalings(features, target_param)

            target = {
                "params": target_param,
                "features": features,
                "feature_loss_functions": feature_loss_functions,
                "feature_start_ends": feature_start_ends,
                "feature_auto_scales": feature_auto_scales,
                "feature_user_scales": feature_user_scales,

                # statistics ...
                "count": 0,
                "feature_losses": [ValueQueue() for _ in features],
                "feature_similarities": [ValueQueue() for _ in features],
            }

            # --- setup transforms ---

            is_random = False  # determine if the transform stack includes randomization
            transforms = []

            for trans_param in target_param["transforms"]:
                for key, transform_param in trans_param.items():
                    if transform_param is None:
                        continue
                    if key not in transform_modules.transformations:
                        raise ValueError(f"Unknown transformation '{key}'")
                    klass = transform_modules.transformations[key]

                    if isinstance(transform_param, dict):
                        t = klass(**transform_param)
                    else:
                        t = klass(transform_param)

                    transforms.append(t)
                    is_random |= klass.IS_RANDOM

            if transforms:
                target["is_random"] = is_random
                target["transforms"] = transforms

            # --- setup constraints ---

            target["constraints"] = []
            for constr_param in target_param["constraints"]:
                for name, Module in constraint_modules.constraints.items():
                    if constr_param.get(name):
                        kwargs = Module.strip_parameters(constr_param[name])
                        constraint = Module(**kwargs)

                        target["constraints"].append({
                            "params": constr_param,
                            "model": constraint.to(self.device),
                            "weight": constr_param[name]["weight"],
                            "start": constr_param[name]["start"],
                            "end": constr_param[name]["end"],
                            # statistics ...
                            "losses": ValueQueue(),
                        })

            self.targets.append(target)

    def _get_features_scalings(self, features: torch.Tensor, target_param: dict) -> torch.Tensor:
        with torch.no_grad():
            #compare_image = torch.zeros([3, 224, 224]).to(features.device) + .1
            compare_image = self.pixel_model.forward()
            if compare_image.shape != torch.Size([3, 224, 224]):
                compare_image = VF.resize(compare_image, [224, 224])
            compare_image = self.clip_preprocess.transforms[-1](compare_image)
            compare_feature = self.clip_model.encode_image(compare_image.unsqueeze(0))
            compare_feature / compare_feature.norm(dim=-1, keepdim=True)
            similarities = (features @ compare_feature.T).squeeze()
            scales = 1. / similarities
            scales /= scales.max()
            return scales

    def initialize(self, context: ExpressionContext):
        self.log(2, "initializing pixels")
        res = context(self.parameters["resolution"])
        self.pixel_model = PixelsRGB(resolution=res).to(self.device)
        self.pixel_model.initialize(self.parameters["init"])
        self.base_learnrate, self.optimizer = \
            create_optimizer(self.pixel_model, self.parameters["optimizer"])

    def _resize_pixel_model(self, res: List[int]):
        self.pixel_model.resize(res)
        self.base_learnrate, self.optimizer = \
            create_optimizer(self.pixel_model, self.parameters["optimizer"])

    def train(self, initialize: bool = True):
        self._running = True
        self._stop = False
        try:
            self._train(initialize=initialize)
        except:
            self._running = False
            raise
        self._running = False

    def _train(self, initialize: bool = True):
        assert self.clip_model

        last_frame_time = None
        last_stats_time = 0
        last_snapshot_time = 0
        last_snapshot_epoch = 0
        loss_queue = ValueQueue()

        epoch_iter = range(self.epoch, self.parameters["epochs"])
        if self.verbose >= 1 and self.log_callback is None:
            epoch_iter = tqdm(epoch_iter)

        self.log(2, "training")
        for epoch in epoch_iter:

            cur_time = time.time()
            if last_frame_time:
                frame_time = cur_time - last_frame_time
                self.training_seconds += frame_time
                self.average_frame_rate += 0.1 * (frame_time - self.average_frame_rate)
            last_frame_time = cur_time

            if self._stop:
                self.log(2, "training stop requested")
                break

            self.epoch = epoch
            self.epoch_f = epoch_f = epoch / max(1, self.parameters["epochs"] - 1)

            def _step(a: float, b: float) -> float:
                inv = False
                if b < a:
                    a, b = b, a
                    inv = True

                if self.epoch_f < a:
                    s = 0.
                elif self.epoch_f > b or a == b:
                    s = 1.
                else:
                    s = (self.epoch_f - a) / (b - a)
                return 1. - s if inv else s

            expression_context = ExpressionContext(
                epoch=epoch,
                time=epoch_f,
                time2=math.pow(epoch_f, 2),
                time3=math.pow(epoch_f, 3),
                time4=math.pow(epoch_f, 4),
                time5=math.pow(epoch_f, 5),
                time_inverse=1. - epoch_f,
                time_inverse2=math.pow(1.-epoch_f, 2),
                time_inverse3=math.pow(1.-epoch_f, 3),
                time_inverse4=math.pow(1.-epoch_f, 4),
                time_inverse5=math.pow(1.-epoch_f, 5),
                time_step=_step,
            )

            if self.pixel_model is None:
                self.initialize(context=expression_context)

            if not self.targets:
                self.setup_targets()

            resolution = expression_context(self.parameters["resolution"])
            if resolution != self.pixel_model.resolution:
                self._resize_pixel_model(resolution)

            expression_context = expression_context.add(
                resolution=[self.pixel_model.pixels.shape[-1], self.pixel_model.pixels.shape[-2]],
                width=self.pixel_model.pixels.shape[-1],
                height=self.pixel_model.pixels.shape[-2],
            )

            # --- update learnrate ---

            learnrate_scale = expression_context(self.parameters["learnrate_scale"])
            learnrate = self.base_learnrate * learnrate_scale * expression_context(self.parameters["learnrate"])

            for g in self.optimizer.param_groups:
                g['lr'] = learnrate

            expression_context = expression_context.add(
                learnrate=learnrate,
                learnrate_scale=learnrate_scale,
            )

            forward_start_time = time.time()

            # --- post process pixels ---

            self._postproc(expression_context)

            # --- get pixels ---

            current_pixels = self.pixel_model.forward()

            # --- apply target transforms ---

            active_targets = []
            target_pixels = []
            target_to_pixels_mapping = dict()
            for target in self.targets:
                if self._check_start_end(
                        target["params"]["start"], target["params"]["end"],
                ):
                    active_targets.append(target)
                    target_idx = len(active_targets) - 1
                    for batch_idx in range(target["params"]["batch_size"]):
                        if not target.get("is_random") and target_idx in target_to_pixels_mapping:
                            continue

                        pixels = current_pixels
                        if target.get("transforms"):
                            for t in target["transforms"]:
                                pixels = t(pixels, expression_context)

                        if pixels.shape != [3, 224, 224]:
                            pixels = VF.resize(pixels, [224, 224])

                        target_pixels.append(pixels.unsqueeze(0))
                        target_pixel_idx = len(target_pixels) - 1

                        if target_idx not in target_to_pixels_mapping:
                            target_to_pixels_mapping[target_idx] = [target_pixel_idx]
                        else:
                            target_to_pixels_mapping[target_idx].append(target_pixel_idx)

            self.forward_seconds += time.time() - forward_start_time

            if active_targets:
                backward_start_time = time.time()

                target_pixels = torch.cat(target_pixels, dim=0)

                # --- retrieve CLIP features ---

                target_pixels = torch.clamp(target_pixels, 0, 1)
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

                self.backward_seconds += time.time() - backward_start_time

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
                mean = value_str(current_pixels.reshape(3, -1).mean(1))
                std = value_str(current_pixels.reshape(3, -1).std(1))
                sat = value_str(get_mean_saturation(current_pixels))
                edge_mean = value_str(constraint_modules.get_edge_mean(current_pixels))
                self.log(2, f"--- train step {epoch+1} / {self.parameters['epochs']} ---")
                self.log(2, f"device: {self.device}")
                self.log(2, f"image: res {self.parameters['resolution']}, "
                            f"mean {mean}, std {std}, sat {sat}, edge_mean {edge_mean}")
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

    def _postproc(self, context: ExpressionContext):
        image = self.pixel_model.pixels
        changed = False

        for pp in self.postprocs:

            if not self._check_start_end(pp["params"]["start"], pp["params"]["end"]):
                continue

            image = pp["transform"](image, context)
            changed = True

        if changed:
            with torch.no_grad():
                self.pixel_model.pixels[...] = image

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

            if target["feature_auto_scales"] is not None:
                similarities *= target["feature_auto_scales"]

            similarities *= torch.Tensor([
                context(w) for w in target["feature_user_scales"]
            ]).to(similarities.device)

            feature_weights: List[Tuple[float, bool]] = self._get_target_feature_weights(target, similarities, context)

            target["applied_feature_weights"] = []

            if target["params"]["select"] != "mix":

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

            # mix features together
            else:
                mixed_features = []
                mixed_weight = 0.
                for i, target_feature in enumerate(target_features):
                    feature_weight, apply_feature = feature_weights[i]
                    if apply_feature:
                        mixed_features.append(feature_weight * target_feature.unsqueeze(0))
                        mixed_weight += abs(feature_weight)

                if mixed_features and mixed_weight:
                    mixed_features = torch.cat(mixed_features)
                    mixed_features = mixed_features.sum(0) / mixed_weight

                    # take the loss function of the first feature
                    loss_function = target["feature_loss_functions"][0]
                    loss = loss_function(clip_feature, mixed_features.squeeze(0))
                    loss_sum += loss

                    for i, target_feature in enumerate(target_features):
                        feature_weight, apply_feature = feature_weights[i]
                        if not apply_feature:
                            target["applied_feature_weights"].append(feature_weight)
                            target["feature_losses"][i].append(0, count=False)
                            target["feature_similarities"][i].append(float(similarities[i]), count=False)
                        else:
                            target["applied_feature_weights"].append(feature_weight)
                            target["feature_losses"][i].append(float(loss))
                            target["feature_similarities"][i].append(float(similarities[i]))
                else:
                    for i, target_feature in enumerate(target_features):
                        feature_weight, apply_feature = feature_weights[i]
                        target["applied_feature_weights"].append(feature_weight)
                        target["feature_losses"][i].append(0, count=False)
                        target["feature_similarities"][i].append(float(similarities[i]), count=False)

            mean_sim = float(similarities.mean())
            context = context.add(similarity=mean_sim)
        else:
            context = context.add(similarity=0.)

        # --- apply constraints ---

        for constraint in target.get("constraints", []):
            if not self._check_start_end(
                    constraint["start"], constraint["end"],
            ):
                continue
            loss = constraint["model"].forward(pixels, context)
            loss = loss * context(constraint["weight"])
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
        for i, target_feature in enumerate(target["features"]):
            feature_context = context.add(similarity=float(similarities[i]))
            weight = feature_context(target["params"]["features"][i]["weight"])
            enable = self._check_start_end(*target["feature_start_ends"][i])
            feature_weights.append([weight, enable])

        if mode in ("all", "mix"):
            pass

        elif mode in ("best", "worst"):
            best_indices = torch.argsort(similarities, descending=mode == "best")
            set_false = False
            for i in best_indices:
                if set_false:
                    feature_weights[i][1] = False

                # catch the best active feature and deactivate all others
                if feature_weights[i][1]:
                    set_false = True

        else:
            raise ValueError(f"Unknown feature selection mode '{mode}'")

        return [tuple(i) for i in feature_weights]

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
            "average_frame_rate": self.average_frame_rate,
            "training_seconds": self.training_seconds,
            "forward_seconds": self.forward_seconds,
            "backward_seconds": self.backward_seconds,
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
                    "weight": round(context(constraint["weight"]), 3),
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
        all_lines = []
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
            all_lines.append(line)

        self.last_target_stats = "\n".join(all_lines)

    def _check_start_end(
            self,
            start: Union[int, float],
            end: Union[int, float],
    ):
        if isinstance(start, int) and start > self.epoch:
            return False
        elif isinstance(start, float) and start > self.epoch_f:
            return False
        if isinstance(end, int) and end < self.epoch:
            return False
        elif isinstance(end, float) and end < self.epoch_f:
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
        return cosine_similarity_loss

    else:
        raise ValueError(f"Invalid loss function '{name}'")


def cosine_similarity_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return 1. - F.cosine_similarity(x1.unsqueeze(0), x2.unsqueeze(0))[0]
