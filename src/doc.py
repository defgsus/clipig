import re
from io import StringIO
from pathlib import Path
import warnings
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
    from .expression import EXPRESSION_ARGS

    for path, param in PARAMETERS.items():
        path: str
        param: Parameter

        is_list = isinstance(param, SequenceParameter)
        is_section = isinstance(param, PlaceholderParameter)
        is_single_param = not any(filter(
            lambda p: p.startswith(path + "."),
            PARAMETERS.keys()
        )) and (
            path.startswith("targets.transforms.") and len(path.split(".")) == 3
        )
        is_new_section = is_section and len(path.split(".")) <= 2

        if is_new_section:
            print("\n\n---\n\n", file=file)

        if is_section:
            print(f"### `{path}`\n", file=file)
            if param.doc:
                print(prepare_doc_string(param.doc) + "\n", file=file)
            else:
                warnings.warn(f"No documentation of '{path}'")
            continue

        if is_single_param:
            print(f"### `{path}`\n", file=file)
        else:
            print(f"#### `{path}`\n", file=file)

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
            default_str = f"default: **`{param.default}`**"

        print(f"`{type_str}` {default_str}\n", file=file)

        if param.expression_groups:
            group_names = [
                EXPRESSION_ARGS[n]["name"]
                for n in sorted(set(param.expression_groups))
            ]
            print("\nexpression variables: " + ", ".join(
                f"[{n}](#{n.replace(' ', '-')}-variable)"
                for n in group_names
            ) + "\n", file=file)

        if param.doc:
            print(prepare_doc_string(param.doc) + "\n", file=file)
        else:
            warnings.warn(f"No documentation of '{path}'")


def prepare_doc_string(doc: str, indent: int = 0) -> str:
    doc = strip_doc(doc)

    links = {
        "CLIPig": "https://github.com/defgsus/CLIPig/",
        "CLIP": "https://github.com/openai/CLIP/",
        "gaussian blur": "https://en.wikipedia.org/wiki/Gaussian_blur",
    }

    def _repl(m):
        key, suffix = m.groups()
        return f"[{key}]({links[key]}){suffix}"

    for key, href in links.items():
        doc = re.sub(f"({key})([\s\-\.'])", _repl, doc)

    if indent:
        doc = "\n".join(
            " " * indent + line
            for line in doc.splitlines()
        )

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


def render_markdown_documentation(template: str) -> str:
    template = prepare_doc_string(template)

    for key, render_func in (
            ("transforms", dump_transforms),
            ("constraints", dump_constraints),
            ("variables", dump_expression_variables),
            ("reference", dump_parameters_md),
    ):
        file = StringIO()
        render_func(file=file)
        file.seek(0)
        text = file.read()

        template = template.replace("{{%s}}" % key, text)

    return template


def dump_constraints(file: Optional[TextIO] = None):
    from .constraints import constraints
    for name in sorted(constraints):
        klass = constraints[name]

        text = klass.__doc__.strip()
        if "\n\n" in text:
            text = text[:text.index("\n\n")]

        print(f"- [{name}](#targetsconstraints{name}): {text}", file=file)


def dump_transforms(file: Optional[TextIO] = None):
    from .transforms import transformations
    for name in sorted(transformations):
        klass = transformations[name]

        text = klass.__doc__.strip()
        if "\n\n" in text:
            text = text[:text.index("\n\n")]

        print(f"- [{name}](#targetstransforms{name}): {text}", file=file)


def dump_expression_variables(file: Optional[TextIO] = None):
    from .expression import EXPRESSION_ARGS
    for group_id, group in EXPRESSION_ARGS.items():

        print(f"### {group['name']} variables\n", file=file)
        print(prepare_doc_string(group["doc"]) + "\n", file=file)

        for variable_name, variable in group["args"].items():
            if not variable.get("doc"):
                continue
            print(f"- #### `{variable_name}` variable\n", file=file)
            print(f"  type: `{variable['type']}`\n", file=file)
            print(prepare_doc_string(variable["doc"], indent=2), file=file)
