import os
import pathlib
import argparse
from io import StringIO
from copy import deepcopy
from typing import Union, Sequence, List, Type, Tuple, Optional, Callable, Any

import yaml

from .files import prepare_output_name
from .expression import Expression


class Parameter:

    def __init__(
            self,
            types: Union[Type, Sequence[Type]],
            default: Any = None,
            null: bool = False,
            doc: Optional[str] = None,
            expression: bool = False,
            expression_args: Optional[Sequence[str]] = None,
    ):
        self.doc = doc
        self.types = list(types) if isinstance(types, Sequence) else [types]
        self.null = null
        self.default = default
        self.expression = expression or bool(expression_args)
        expression_args = expression_args or EXPR_ARGS.DEFAULT
        self.expression_args = list(expression_args)

        assert len(self.types)
        assert len(self.types) == 1 or not self.expression

    def convert(self, x: Any) -> Any:
        if self.null and x is None:
            return None

        for t in self.types:
            if t == type(x):
                return x

        for t in self.types:
            try:
                return t(x)
            except (TypeError, ValueError):
                pass

        if not self.expression:
            if len(self.types) == 1:
                raise ValueError(
                    f"Expected type '{self.types[0]}', got '{x}'"
                )
            else:
                raise ValueError(
                    f"Expected one of types {self.types}, got '{x}'"
                )

        exp = Expression(self.types[0], x, *self.expression_args)
        exp.validate()
        return exp


class FrameTimeParameter(Parameter):

    def __init__(
            self,
            default,
            doc: Optional[str] = None,
    ):
        super().__init__(
            types=[int, float],
            default=default,
            doc=doc
        )

    def convert(self, x: Any) -> Any:
        factor = None
        if isinstance(x, str):
            if x.endswith("%"):
                x = x[:-1]
                factor = 1. / 100.

        value = super().convert(x)

        if factor is not None:
            value *= factor
        return value


class SequenceParameter(Parameter):
    def __init__(
            self,
            types: Union[Type, Sequence[Type]],
            length: int,
            default: Any = None,
            null: bool = False,
            doc: Optional[str] = None,
            expression: bool = False,
            expression_args: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            types=types,
            default=default,
            null=null,
            doc=doc,
            expression=expression,
            expression_args=expression_args,
        )
        self.length = length

    def convert(self, x: Any) -> Any:
        if self.null and x is None:
            return None

        if isinstance(x, (list, tuple)):
            sequence = list(x)
        elif isinstance(x, str):
            sequence = x.split() if "," not in x else x.split(",")
        else:
            sequence = [x]

        if len(sequence) == 1 and self.length > 1:
            sequence = sequence * self.length
        elif len(sequence) == self.length:
            pass
        else:
            length_str = "1"
            if self.length > 1:
                length_str += f" or {self.length}"
            raise ValueError(f"expected list of length {length_str}, got {len(sequence)}")

        return [
            super(SequenceParameter, self).convert(i)
            for i in sequence
        ]


class PlaceholderParameter(Parameter):
    """
    Just a placeholder to open new sub-parameter lists or dicts
    """
    pass


class EXPR_ARGS:

    MINIMAL = (
        "epoch", "t",
    )

    LEARNRATE = (
        "lr", "learnrate",
        "lrs", "learnrate_scale",
    )

    DEFAULT = MINIMAL + LEARNRATE

    TARGET_FEATURE = DEFAULT + (
        "sim", "similarity",
    )

    TARGET_CONSTRAINT = DEFAULT + (
        "sim", "similarity",
    )


PARAMETERS = {
    "verbose": Parameter(int, default=2),
    "snapshot_interval": Parameter([int, float], default=20.),
    "device": Parameter(str, default="auto"),
    "learnrate": Parameter(float, default=1., expression=True, expression_args=EXPR_ARGS.MINIMAL),
    "learnrate_scale": Parameter(float, default=1., expression=True, expression_args=EXPR_ARGS.MINIMAL),
    "output": Parameter(str, default=f".{os.path.sep}"),
    "epochs": Parameter(int, default=300),
    "start_epoch": Parameter(int, default=0),
    "resolution": SequenceParameter(int, length=2, default=[224, 224]),
    "model": Parameter(str, default="ViT-B/32"),
    "optimizer": Parameter(str, default="adam"),

    "init": PlaceholderParameter(dict, default=dict()),
    "init.mean": SequenceParameter(float, length=3, default=[.5, .5, .5]),
    "init.std": SequenceParameter(float, length=3, default=[.1, .1, .1]),
    "init.image": Parameter(str, null=True, default=None),
    "init.image_tensor": Parameter(list, default=None),

    "postproc": PlaceholderParameter(list, default=list()),
    "postproc.active": Parameter(bool, default=True),
    "postproc.start": FrameTimeParameter(default=0.0),
    "postproc.end": FrameTimeParameter(default=1.0),
    "postproc.blur.kernel_size": SequenceParameter(int, length=2, default=[3, 3]),
    "postproc.blur.sigma": SequenceParameter(float, length=2, null=True, default=None),
    "postproc.add": SequenceParameter(float, length=3, expression=True),
    "postproc.multiply": SequenceParameter(float, length=3, expression=True),

    "targets": PlaceholderParameter(list, default=list()),
    "targets.active": Parameter(bool, default=True),
    "targets.name": Parameter(str, default="target"),
    "targets.start": FrameTimeParameter(default=0.0),
    "targets.end": FrameTimeParameter(default=1.0),
    "targets.weight": Parameter(float, default=1., expression=True, expression_args=EXPR_ARGS.TARGET_CONSTRAINT),
    "targets.batch_size": Parameter(int, default=1),
    "targets.select": Parameter(str, default="all"),
    "targets.features": PlaceholderParameter(list, default=list()),
    "targets.features.weight": Parameter(float, default=1., expression=True, expression_args=EXPR_ARGS.TARGET_FEATURE),
    "targets.features.loss": Parameter(str, default="cosine"),
    "targets.features.text": Parameter(str, null=True),
    "targets.features.image": Parameter(str, null=True),
}


def _add_parameters(prefix: str, classes: dict):
    for name, klass in classes.items():
        params = klass.PARAMS
        if len(params) == 1:
            PARAMETERS[f"{prefix}.{name}"] = next(iter(params.values()))
        else:
            for param_name, value in params.items():
                PARAMETERS[f"{prefix}.{name}.{param_name}"] = value


def _add_class_parameters():
    from .transforms import transformations
    from .constraints import constraints
    PARAMETERS["targets.transforms"] = PlaceholderParameter(list, default=list())
    _add_parameters("targets.transforms", transformations)

    PARAMETERS["targets.constraints"] = PlaceholderParameter(list, default=list())
    _add_parameters("targets.constraints", constraints)


_add_class_parameters()


def parse_arguments(gui_mode: bool = False) -> dict:
    """
    Returns full set of parameters to run the experiment.

    Parameters from yaml files and command-line parameters are merged

    :param gui_mode: bool,
        If False, at least one config file is required
        and parameters get converted and defaults are added
    :return: dict
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, nargs="*" if gui_mode else "+", default=[],
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
        help="Learnrate scaling factor, defaults to %s" % PARAMETERS["learnrate"].default,
    )
    parser.add_argument(
        "-opt", "--optimizer", type=str, default=None,
        help="Optimizer that performs the gradient descent, defaults to %s" % PARAMETERS["optimizer"].default,
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=None,
        help="Number of training steps, default = %s" % PARAMETERS["epochs"].default,
    )
    parser.add_argument(
        "-r", "--resolution", type=int, default=None, nargs="+",
        help="Resolution in pixels, can be one or two numbers, "
             "defaults to %s" % PARAMETERS["resolution"].default,
    )
    parser.add_argument(
        "-s", "--snapshot-interval", type=float, default=None,
        help="Number of seconds after which a snapshot is saved, "
             "defaults to %s" % PARAMETERS["snapshot_interval"].default,
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=None,
        help="Verbosity. Default is %s" % PARAMETERS["verbose"].default,
    )
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="Device to run on, either 'auto', 'cuda' or 'cuda:1', etc... "
             "Default is %s" % PARAMETERS["device"].default,
    )

    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to run",
    )

    args = parser.parse_args()

    output_name = ""
    parameters = dict()

    for filename in args.config:
        yaml_config = load_yaml_config(filename, convert=not gui_mode)
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

    for key in ("optimizer", "learnrate", "epochs", "resolution", "device", "snapshot_interval"):
        if getattr(args, key) is not None:
            parameters[key] = getattr(args, key)

    if args.output is not None:
        if args.output.endswith(os.path.sep):
            output_name = os.path.join(args.output, output_name.split(os.path.sep)[-1])
        else:
            output_name = args.output

    if not gui_mode:
        parameters = convert_params(parameters)
        set_parameter_defaults(parameters)

    if parameters:
        parameters["output"] = str(prepare_output_name(output_name, make_dir=False))

    return parameters


def load_yaml_config(filename: str, convert: bool = True) -> dict:
    try:
        with open(filename) as fp:
            params = yaml.safe_load(fp)
            if convert:
                params = convert_params(params)
            return params
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


def parameters_to_yaml(parameters: dict) -> str:
    file = StringIO()
    yaml.safe_dump(parameters, file, sort_keys=False)
    file.seek(0)
    return file.read()


def yaml_to_parameters(text: str) -> dict:
    file = StringIO(text)
    return yaml.safe_load(file)


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
            if isinstance(value, list) and isinstance(PARAMETERS[path], PlaceholderParameter):
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
        param: Parameter = PARAMETERS[path]

        try:
            if param.expression and isinstance(value, Expression):
                return value
            return param.convert(value)
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
            params[param_name] = deepcopy(param_info.default)

        if isinstance(param_info, PlaceholderParameter):
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
