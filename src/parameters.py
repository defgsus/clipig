import os
import pathlib
import argparse
from io import StringIO
from copy import deepcopy
from typing import Union, Sequence, List, Type, Tuple, Optional, Callable, Any

import yaml
import clip

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

    def copy(self):
        return self.__class__(
            types=self.types,
            default=self.default,
            null=self.null,
            doc=self.doc,
            expression=self.expression,
            expression_args=self.expression_args,
        )

    def convert(self, x: Any) -> Any:
        if self.null and x is None:
            return None

        if isinstance(x, Expression):
            return x

        # type matches?
        for t in self.types:
            if t == type(x):
                return x

        # is convertible to type?
        for t in self.types:
            try:
                return t(x)
            except (TypeError, ValueError):
                pass

        # maybe an expression without variables?
        if isinstance(x, str):
            try:
                exp = Expression(self.types[-1], x)
                return self.types[-1](exp())
            except:
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

    def copy(self):
        return self.__class__(
            types=self.types,
            length=self.length,
            default=self.default,
            null=self.null,
            doc=self.doc,
            expression=self.expression,
            expression_args=self.expression_args,
        )

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
        "epoch",
        "t", "t2", "t3", "t4", "t5",
        "ti", "ti2", "ti3", "ti4", "ti5",
    )

    LEARNRATE = (
        "lr", "learnrate",
        "lrs", "learnrate_scale",
    )

    DEFAULT = MINIMAL + LEARNRATE

    TARGET_TRANSFORM = DEFAULT

    TARGET_FEATURE = DEFAULT + (
        "sim", "similarity",
    )

    TARGET_CONSTRAINT = DEFAULT + (
        "sim", "similarity",
    )


PARAMETERS = {
    "verbose": Parameter(
        int, default=2,
        doc="""
        Verbosity level
        - `0` = off
        - `1` = show progress
        - `2` = show statistics
        """
    ),
    "output": Parameter(
        str, default=f".{os.path.sep}",
        doc=f"""
        Directory or filename of the output. 
        - If a directory, it must end with `{os.path.sep}`. 
          In that case, the filename will be the name of the yaml config file. 
        - If a filename, it must end with `.png`. Note that a number is attached to the 
          filename or is automatically increased, if the file already exists.
        """
    ),
    "snapshot_interval": Parameter(
        [int, float], default=20.,
        doc="""
        Interval after which a snapshot of the currently trained image is saved. 
        
        A float number specifies the interval in seconds. An integer number specifies 
        the interval in number-of-epochs.
        """
    ),
    "epochs": Parameter(
        int, default=300,
        doc="""
        The number of training steps before stopping the training, not including batch sizes. 
        
        For example, if the number of epochs is `100` and a target has a batch_size of `10`, 
        then `1000` training steps will be performed.
        """
    ),
    "start_epoch": Parameter(
        int, default=0,
        doc="""
        The number of epochs to skip in the beginning. 
        
        This is used by the GUI application to continue training after config changes.
        """
    ),
    "resolution": SequenceParameter(
        int, length=2, default=[224, 224],
        doc="Resolution of the image to create. A single number for square images or two "
            "numbers for width and height."
    ),
    "model": Parameter(
        str, default="ViT-B/32",
        doc=("The pre-trained CLIP model to use. Options are " +
             ", ".join(f"`{m}`" for m in clip.available_models()) +
             "\n\nThe models are downloaded from `openaipublic.azureedge.net` and stored "
             "in the current user's cache directory"
             )
    ),
    "device": Parameter(
        str, default="auto",
        doc="The device to run the training on. Can be `cpu`, `cuda`, `cuda:1` etc.",
    ),
    "learnrate": Parameter(
        float, default=1., expression_args=EXPR_ARGS.MINIMAL,
        doc="""
        The learning rate of the optimizer. 
        
        Different optimizers have different learning rates that work well. 
        However, this value is scaled *by hand* so that `1.0` translates to 
        about the same learning rate for each optimizer. 
        
        The learnrate value is available to other expressions as `lr` or `learnrate`.
        """
    ),
    "learnrate_scale": Parameter(
        float, default=1., expression_args=EXPR_ARGS.MINIMAL,
        doc="""
        A scaling parameter for the actual learning rate.
        
        It's for convenience in the case when learnrate_scale is an expression like `1. - t`. 
        The actual learnrate can be overridden with fixed values like `2` or `3` in 
        different experiments.
        
        The learnrate_scale value is available to other expressions as `lrs` or `learnrate_scale`.
        """
    ),
    "optimizer": Parameter(
        str, default="adam",
        doc="The torch optimizer to perform the gradient descent."
    ),

    "init": PlaceholderParameter(
        dict, default=dict(),
        doc="""
        Defines the way, the pixels are initialized. Default is random pixels.
        """
    ),
    "init.mean": SequenceParameter(
        float, length=3, default=[.5, .5, .5],
        doc="""
        The mean (brightness) of the initial pixel noise. 
        
        Can be a single number for gray or three numbers for RGB.
        """
    ),
    "init.std": SequenceParameter(
        float, length=3, default=[.1, .1, .1],
        doc="""
        The standard deviation (randomness) of the initial pixel noise. 
        
        A single number will be copied to the RGB values.
        """
    ),
    "init.image": Parameter(
        str, null=True, default=None,
        doc="""
        A filename of an image to use as starting point.
        
        The image will be scaled to the desired resolution if necessary.
        """
    ),
    "init.image_tensor": Parameter(
        list, null=True, default=None,
        doc="""
        A 3-dimensional matrix of pixel values in the range [0, 1]  
        
        The layout is the same as used in 
        [torchvision](https://pytorch.org/vision/stable/index.html), 
        namely `[C, H, W]`, where `C` is number of colors (3), 
        `H` is height and `W` is width.
        
        This is used by the GUI application to continue training after config changes.
        """
    ),

    "targets": PlaceholderParameter(
        list, default=list(),
        doc="""
        This is a list of *targets* that define the desired image. 
        
        Most important are the [features](#targetsfeatures) where
        texts or images are defined which get converted into CLIP
        features and then drive the image creation process.
        
        It's possible to add additional [constraints](#targetsconstraints)
        which alter image creation without using CLIP, 
        e.g. the image mean, saturation or gaussian blur.
        """
    ),
    "targets.active": Parameter(
        bool, default=True,
        doc="""
        A boolean to turn of the target during development. 
        
        This is just a convenience parameter. To turn of a target
        during testing without deleting all the parameters, simply 
        put `active: false` inside.
        """
    ),
    "targets.name": Parameter(
        str, default="target",
        doc="""
        The name of the target. 
        
        Currently this is just displayed in the statistics dump and has no
        functionality. 
        """
    ),
    "targets.start": FrameTimeParameter(
        default=0.0,
        doc="""
        Start frame of the target. 
        
        - an `int` number defines the time as epoch frame
        - a `float` number defines the time as ratio between 0.0 and 1.0, 
          where 1.0 is the final epoch.
        - `percent` (e.g. `23.5%`) defines the time as percentage of the number of epochs. 
        """
    ),
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


def _add_parameters(prefix: str, classes: dict, expr_args: Tuple[str, ...] = None):
    def _add_args(p: Parameter) -> Parameter:
        if not expr_args:
            return p
        p = p.copy()
        p.expression = True
        p.expression_args = expr_args
        return p

    for name in sorted(classes.keys()):
        params = classes[name].PARAMS
        if len(params) == 1:
            PARAMETERS[f"{prefix}.{name}"] = _add_args(next(iter(params.values())))
        else:
            PARAMETERS[f"{prefix}.{name}"] = PlaceholderParameter(dict, default=None)
            for param_name, value in params.items():
                PARAMETERS[f"{prefix}.{name}.{param_name}"] = _add_args(value)


def _add_class_parameters():
    from .transforms import transformations
    from .constraints import constraints
    PARAMETERS["targets.transforms"] = PlaceholderParameter(list, default=list())
    _add_parameters("targets.transforms", transformations, expr_args=EXPR_ARGS.TARGET_TRANSFORM)

    PARAMETERS["targets.constraints"] = PlaceholderParameter(list, default=list())
    _add_parameters("targets.constraints", constraints, expr_args=EXPR_ARGS.TARGET_CONSTRAINT)

    postprocs = {
        name: klass
        for name, klass in transformations.items()
        if not klass.IS_RESIZE
    }
    PARAMETERS.update({
        "postproc": PlaceholderParameter(list, default=list()),
        "postproc.active": Parameter(bool, default=True),
        "postproc.start": FrameTimeParameter(default=0.0),
        "postproc.end": FrameTimeParameter(default=1.0),
    })
    _add_parameters("postproc", postprocs, expr_args=EXPR_ARGS.DEFAULT)


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
        "config", type=str, nargs="*", default=[],
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
        "-d", "--device", type=str, default=None,
        help="Device to run on, either 'auto', 'cuda' or 'cuda:1', etc... "
             "Default is %s" % PARAMETERS["device"].default,
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to run",
    )
    parser.add_argument(
        "-v", "--verbose", type=int, default=None,
        help="Verbosity. Default is %s" % PARAMETERS["verbose"].default,
    )

    args = parser.parse_args()

    if not args.config and not gui_mode:
        from .doc import dump_parameters_text
        dump_parameters_text(PARAMETERS)
        print("\nPlease specify a yaml configuration file")
        exit(-1)

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
