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


def dump_parameters_md(file: Optional[TextIO] = None):
    from .parameters import PARAMETERS, Parameter, SequenceParameter, PlaceholderParameter

    for path, param in PARAMETERS.items():
        path: str
        param: Parameter

        is_list = isinstance(param, SequenceParameter)
        is_section = isinstance(param, PlaceholderParameter)

        if is_section:
            print(f"### {path}\n", file=file)
            if param.doc:
                print(strip_doc(param.doc) + "\n", file=file)
            continue

        print(f"#### {path}\n", file=file)

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

        if param.default is None:
            default_str = "no default"
        else:
            default_str = f"default: **{param.default}**"

        print(f"`{type_str}` {default_str}\n", file=file)

        if param.doc:
            print(strip_doc(param.doc) + "\n", file=file)


def prepare_doc_string(doc: str) -> str:
    # [CLIP](https://github.com/openai/CLIP/)
    return doc


def strip_doc(doc: Optional[str]) -> Optional[str]:
    if not doc:
        return doc

    min_lstrip = min(
        len(line) - len(line.lstrip())
        for line in doc.splitlines()
        if line.strip()
    )

    doc = "\n".join(
        line[min_lstrip:]
        for line in doc.splitlines()
    )

    return doc.strip()
