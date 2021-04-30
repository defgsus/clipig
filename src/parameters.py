import math
import random
import traceback
import argparse
import time
import json
from copy import deepcopy
from typing import Union, Sequence, Type, Tuple, Optional, Callable

import yaml


def sequence_converter(Type: Callable, length: int) -> Callable:
    def _convert(v):
        if isinstance(v, (list, tuple)):
            sequence = list(v)
        elif isinstance(v, Type):
            sequence = [v]
        elif isinstance(v, str):
            sequence = [Type(i) for i in v.split()]
        else:
            try:
                v = Type(v)
                sequence = [v]
            except:
                raise TypeError(f"expected type {Type.__name__}, got {type(v).__name__}")

        if len(sequence) == 1:
            sequence = sequence * length
        elif len(sequence) == length:
            pass
        else:
            length_str = "1"
            if length > 1:
                length_str += f" or {length}"
            raise ValueError(f"expected list of length {length_str}, got {len(sequence)}")

        return sequence
    return _convert


def frame_time_converter(v):
    if isinstance(v, str):
        if v.endswith("%"):
            v = float(v[:-1]) / 100.
        else:
            try:
                v = int(v)
            except ValueError:
                v = float(v)

    if isinstance(v, (int, float)):
        return v

    raise TypeError(f"expected int, float or percent, got {type(v).__name__}")


PARAMETERS = {
    "learnrate": {"convert": float, "default": 1.},
    "epochs": {"convert": int, "default": 300},
    "resolution": {"convert": sequence_converter(int, 2), "default": [224, 244]},
    "init": {"default": dict()},
    "init.mean": {"convert": sequence_converter(float, 3), "default": [.5, .5, .5]},
    "init.std": {"convert": sequence_converter(float, 3), "default": [.1, .1, .1]},
    "targets": {"default": list()},
    "targets.name": {"convert": str, "default": "target"},
    "targets.start": {"convert": frame_time_converter, "default": 0.0},
    "targets.end": {"convert": frame_time_converter, "default": 1.0},
    "targets.weight": {"convert": float, "default": 1.0},
    "targets.mean_saturation_max": {"convert": float, "default": None},
    "targets.features": {"default": list()},
    "targets.features.weight": {"convert": float, "default": 1.0},
    "targets.features.text": {"convert": str, "default": None},
    "targets.features.image": {"convert": str, "default": None},
    "targets.transforms": {"default": list()},
    "targets.transforms.translate": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.resize": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.rotate": {"convert": sequence_converter(float, 2), "default": None},
}


def parse_arguments() -> dict:
    """
    Returns full set of parameters to run the experiment.

    Parameters from yaml files and command-line parameters are merged

    :return: dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, nargs="+", default=[],
        help="Configuration yaml file(s). When several files, "
             "parameters will be merged together where later config files "
             "will overwrite previous parameters. All other command line "
             "arguments will overwrite the yaml parameters.",
    )
    parser.add_argument(
        "-lr", "--learnrate", type=float, default=PARAMETERS["learnrate"]["default"],
        help="Learnrate scaling factor, defaults to %s" % PARAMETERS["learnrate"]["default"],
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=PARAMETERS["epochs"]["default"],
        help="Number of training steps, default = %s" % PARAMETERS["epochs"]["default"],
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=PARAMETERS["resolution"]["default"], nargs="+",
        help="Resolution in pixels, can be one or two numbers, "
             "defaults to %s" % PARAMETERS["resolution"]["default"],
    )

    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to run",
    )

    args = parser.parse_args()


    parameters = dict()

    for filename in args.config:
        parameters = merge_parameters(
            parameters,
            load_yaml_config(filename),
        )

    parameters.update({
        "learnrate": args.learnrate,
        "epochs": args.epochs,
        "resolution": args.resolution,
    })
    parameters = convert_params(parameters)
    set_parameter_defaults(parameters)

    return parameters


def load_yaml_config(filename: str) -> dict:
    try:
        with open(filename) as fp:
            return convert_params(yaml.safe_load(fp.read()))
    except Exception as e:
        e.args = (f"{filename}: {e}", )
        raise


def convert_params(data: dict) -> dict:
    return _recursive_convert_and_validate(data, "")


def _recursive_convert_and_validate(data, parent_path: str):
    if isinstance(data, dict):
        data = deepcopy(data)
        for key, value in data.items():
            path = key if not parent_path else f"{parent_path}.{key}"
            if isinstance(value, list) and not PARAMETERS[path].get("convert"):
                for i, v in enumerate(value):
                    value[i] = _recursive_convert_and_validate(v, path)
            else:
                data[key] = _recursive_convert_and_validate(value, path)

        return data
    else:
        path = parent_path
        value = data

        if path not in PARAMETERS:
            raise ValueError(f"unknown parameter '{path}'")
        param = PARAMETERS[path]

        try:
            return param["convert"](value)
        except Exception as e:
            e.args = (f"parameter '{path}': {e}", )
            raise


def merge_parameters(params1: dict, params2: dict) -> dict:
    """
    Merge parameters 1 and 2, while 2 always wins
    :return: new merged dict
    """
    return _recursive_merge_parameters(params1, params2)


def _recursive_merge_parameters(params1: dict, params2: dict) -> dict:
    params1 = deepcopy(params1)
    for key, value in params2.items():

        if isinstance(value, dict):
            if key not in params1 or not params1[key]:
                params1[key] = dict()
            params1[key] = _recursive_merge_parameters(params1[key], value)

        elif isinstance(value, list):
            # TODO: should be able to merge with target of the same 'name'
            if key not in params1 or not params1[key]:
                params1[key] = list()
            params1[key] += value

        else:
            params1[key] = value

    return params1


def set_parameter_defaults(params: dict):
    _recursive_parameter_defaults(params, "")


def _recursive_parameter_defaults(params: dict, parent_path: str):
    for param_path, param_info in PARAMETERS.items():
        if parent_path and not param_path.startswith(parent_path):
            continue
        param_path = param_path[len(parent_path):].split(".")
        if not param_path[0]:
            param_path = param_path[1:]

        if len(param_path) != 1:
            continue
        param_name = param_path[0]

        if param_name not in params:
            params[param_name] = deepcopy(param_info["default"])

        if not param_info.get("convert"):
            sub_params = params[param_name]
            path = param_name if not parent_path else f"{parent_path}.{param_name}"
            if isinstance(sub_params, list):
                for i, v in enumerate(sub_params):
                    _recursive_parameter_defaults(sub_params[i], path)
            elif isinstance(sub_params, dict):
                _recursive_parameter_defaults(sub_params, path)
            else:
                raise TypeError(f"{path}: unhandled sub-parameter type '{type(sub_params).__name__}'")
