import pathlib
import subprocess
from typing import List, Optional


def render(
        source: str,
        output_name: str,
        template_context: dict,
        snapshot_interval: float = 0.,
        device: str = "auto",
        extra_args: Optional[List[str]] = None,
):
    """
    Render the yaml-config from source via clipig.py

    :param source: str, either a filename or the yaml text
    :param output_name: complete filename of the output image file
    :param template_context: dict with key/value mapping
        where each instane of `{{key}}` will be replaced with `value`.
    :param extra_args: list of str, Extra command-line arguments
    """
    config_text = yaml_template(source, template_context)

    call_args = [
        "python", "clipig.py", "-", "--output", output_name,
        "--snapshot-interval", str(snapshot_interval),
        "--device", device,
    ]
    if extra_args:
        call_args += extra_args

    process = subprocess.Popen(call_args, stdin=subprocess.PIPE)

    process.communicate(config_text.encode("utf-8"))


def yaml_template(
        source: str,
        template_context: dict,
) -> str:
    try:
        text = pathlib.Path(source).read_text()
    except (OSError, IOError):
        text = source

    for key, value in template_context.items():
        text = text.replace("{{%s}}" % key, value)

    return text


