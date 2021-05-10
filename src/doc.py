from typing import TextIO, Optional


def dump_parameters_text(PARAMETERS: dict, file: Optional[TextIO] = None):
    from .parameters import Parameter, SequenceParameter, PlaceholderParameter

    for path, param in PARAMETERS.items():
        path: str
        param: Parameter

        is_list = isinstance(param, SequenceParameter)

        if len(param.types) == 1:
            type_str = param.types[0].__name__
            if is_list:
                type_str = f"of {type_str}"
        else:
            type_str = ", ".join(t.__name__ for t in param.types)
            if is_list:
                type_str = f"of one of {type_str}"

        if is_list:
            type_str = f"list of length {param.length} {type_str}"
        print(f"{path} (type: {type_str})", file=file)
