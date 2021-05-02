import os
import pathlib
import argparse
from copy import deepcopy
from typing import Union, Sequence, Type, Tuple, Optional, Callable

import yaml

from .files import prepare_output_name
from .expression import Expression


class EXPR_ARGS:

    LEARNRATE = (
        "lr", "learnrate",
        "lrs", "learnrate_scale",
    )

    DEFAULT = (
        "epoch", "epoch_f",
        "t",
    ) + LEARNRATE


    TARGET_FEATURE = DEFAULT + (
        "sim", "similarity",
    )

    TARGET_CONSTRAINT = DEFAULT + (
        "sim", "similarity",
    )


def expression_converter(
        type: Type,
        *arguments: str,
        remove: Optional[Sequence[str]] = None
) -> Callable:
    if not arguments:
        arguments = EXPR_ARGS.DEFAULT

    if remove:
        arguments = list(set(arguments) - set(remove))

    def _convert(text):
        return Expression(text, *arguments)
    _convert.is_expression = True

    return _convert


def sequence_converter(
        type_: Callable,
        length: int,
        expr: bool = False,
        expression_args: Sequence[str] = tuple(),
) -> Callable:
    if not expression_args:
        expression_args = EXPR_ARGS.DEFAULT

    def _convert(v):
        if isinstance(v, (list, tuple)):
            sequence = list(v)
        elif isinstance(v, str):
            sequence = v.split() if "," not in v else v.split(",")
        else:
            sequence = [v]

        if len(sequence) == 1:
            sequence = sequence * length
        elif len(sequence) == length:
            pass
        else:
            length_str = "1"
            if length > 1:
                length_str += f" or {length}"
            raise ValueError(f"expected list of length {length_str}, got {len(sequence)}")

        for i, v in enumerate(sequence):
            if isinstance(v, type_):
                continue

            elif isinstance(v, str):
                try:
                    v = type_(v)
                except:
                    if expr:
                        v = Expression(v, *expression_args)
                    else:
                        raise TypeError(f"expected type {type_.__name__}, got {type(v).__name__}")

            sequence[i] = v

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
    "verbose": {"convert": int, "default": 2},
    "device": {"convert": str, "default": "auto"},
    "learnrate": {"convert": expression_converter(float, remove=EXPR_ARGS.LEARNRATE), "default": 1.},
    "learnrate_scale": {"convert": expression_converter(float, remove=EXPR_ARGS.LEARNRATE), "default": 1.},
    "output": {"convert": str, "default": f".{os.path.sep}"},
    "epochs": {"convert": int, "default": 300},
    "resolution": {"convert": sequence_converter(int, 2), "default": [224, 224]},
    "model": {"convert": str, "default": "ViT-B/32"},
    "init": {"default": dict()},
    "init.mean": {"convert": sequence_converter(float, 3), "default": [.5, .5, .5]},
    "init.std": {"convert": sequence_converter(float, 3), "default": [.1, .1, .1]},
    "init.image": {"convert": str, "default": None},
    "postproc": {"default": list()},
    "postproc.active": {"convert": bool, "default": True},
    "postproc.start": {"convert": frame_time_converter, "default": 0.0},
    "postproc.end": {"convert": frame_time_converter, "default": 1.0},
    "postproc.blur": {"convert": sequence_converter(float, 2, expr=True), "default": None},
    "postproc.add": {"convert": sequence_converter(float, 3, expr=True), "default": None},
    "postproc.multiply": {"convert": sequence_converter(float, 3, expr=True), "default": None},
    "targets": {"default": list()},
    "targets.active": {"convert": bool, "default": True},
    "targets.name": {"convert": str, "default": "target"},
    "targets.start": {"convert": frame_time_converter, "default": 0.0},
    "targets.end": {"convert": frame_time_converter, "default": 1.0},
    "targets.weight": {"convert": expression_converter(float), "default": 1.0},
    "targets.select": {"convert": str, "default": "all"},
    "targets.features": {"default": list()},
    "targets.features.weight": {"convert": expression_converter(float, *EXPR_ARGS.TARGET_FEATURE), "default": 1.0},
    "targets.features.loss": {"convert": str, "default": "cosine"},
    "targets.features.text": {"convert": str, "default": None},
    "targets.features.image": {"convert": str, "default": None},
    "targets.constraints": {"default": list()},
    "targets.constraints.mean": {"default": None},
    "targets.constraints.mean.weight": {"convert": expression_converter(float, *EXPR_ARGS.TARGET_CONSTRAINT), "default": 1.},
    "targets.constraints.mean.above": {"convert": sequence_converter(float, 3, expr=True, expression_args=EXPR_ARGS.TARGET_CONSTRAINT), "default": None},
    "targets.constraints.mean.below": {"convert": sequence_converter(float, 3, expr=True, expression_args=EXPR_ARGS.TARGET_CONSTRAINT), "default": None},
    "targets.constraints.std": {"default": None},
    "targets.constraints.std.weight": {"convert": expression_converter(float, *EXPR_ARGS.TARGET_CONSTRAINT), "default": 1.},
    "targets.constraints.std.above": {"convert": sequence_converter(float, 3, expr=True, expression_args=EXPR_ARGS.TARGET_CONSTRAINT), "default": None},
    "targets.constraints.std.below": {"convert": sequence_converter(float, 3, expr=True, expression_args=EXPR_ARGS.TARGET_CONSTRAINT), "default": None},
    "targets.transforms": {"default": list()},
    "targets.transforms.noise": {"convert": sequence_converter(float, 3), "default": None},
    "targets.transforms.blur": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.repeat": {"convert": sequence_converter(int, 2), "default": None},
    "targets.transforms.resize": {"convert": sequence_converter(int, 2), "default": None},
    "targets.transforms.center_crop": {"convert": sequence_converter(int, 2), "default": None},
    "targets.transforms.random_translate": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.random_scale": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.random_rotate.degree": {"convert": sequence_converter(float, 2), "default": [-5, 5]},
    "targets.transforms.random_rotate.center": {"convert": sequence_converter(float, 2), "default": None},
    "targets.transforms.random_crop": {"convert": sequence_converter(int, 2), "default": None},
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
        "-o", "--output", type=str, default=None,
        help="Path with or without filename of the output image. "
             "Defaults to the name of the last specified config file.",
    )
    parser.add_argument(
        "-lr", "--learnrate", type=float, default=None,
        help="Learnrate scaling factor, defaults to %s" % PARAMETERS["learnrate"]["default"],
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=None,
        help="Number of training steps, default = %s" % PARAMETERS["epochs"]["default"],
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=None, nargs="+",
        help="Resolution in pixels, can be one or two numbers, "
             "defaults to %s" % PARAMETERS["resolution"]["default"],
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=None,
        help="Verbosity. Default is %s" % PARAMETERS["verbose"]["default"],
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Device to run on, either 'auto', 'cuda' or 'cuda:1', etc... "
             "Default is %s" % PARAMETERS["device"]["default"],
    )

    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to run",
    )

    args = parser.parse_args()

    output_name = ""
    parameters = dict()

    for filename in args.config:
        yaml_config = load_yaml_config(filename)
        parameters = merge_parameters(parameters, yaml_config)

        output_name = yaml_config.get("output") or ""
        if not output_name or output_name.endswith(os.path.sep):
            if not output_name:
                output_name = f".{os.path.sep}"
            output_name += pathlib.Path(filename).name
            if "." in output_name:
                output_name = ".".join(output_name.split(".")[:-1])
            output_name += ".png"

        elif "." not in output_name:
            output_name += ".png"

    for key in ("learnrate", "epochs", "resolution", "device"):
        if getattr(args, key) is not None:
            parameters[key] = getattr(args, key)

    if args.output is not None:
        if args.output.endswith(os.path.sep):
            output_name = os.path.join(args.output, output_name.split(os.path.sep)[-1])
        else:
            output_name = args.output

    parameters = convert_params(parameters)
    set_parameter_defaults(parameters)

    parameters["output"] = str(prepare_output_name(output_name, make_dir=False))

    return parameters


def load_yaml_config(filename: str) -> dict:
    try:
        with open(filename) as fp:
            return convert_params(yaml.safe_load(fp))
    except Exception as e:
        e.args = (f"{filename}: {e}", )
        raise


def save_yaml_config(
        filename: str,
        parameters: dict,
        header: Optional[str] = None,
        footer: Optional[str] = None,
):
    data = _recursive_export_ready(parameters)
    with open(filename, "w") as fp:
        if header:
            fp.write(header)
        if not header.endswith("\n"):
            fp.write("\n")

        yaml.safe_dump(data, fp, sort_keys=False)

        if footer:
            fp.write(footer)


def _recursive_export_ready(data):
    if isinstance(data, dict):
        new_data = dict()
        for key, value in data.items():
            value = _recursive_export_ready(value)
            if value is None or value == []:
                continue
            new_data[key] = value
        return new_data

    elif isinstance(data, list):
        return [
            _recursive_export_ready(i)
            for i in data
        ]

    elif isinstance(data, Expression):
        return data.expression

    else:
        return data




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
            if getattr(param["convert"], "is_expression", False) and isinstance(value, Expression):
                return value
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
            elif sub_params is None:
                pass
            else:
                raise TypeError(f"{path}: unhandled sub-parameter type '{type(sub_params).__name__}'")
