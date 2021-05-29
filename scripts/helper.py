import pathlib
import subprocess


def render(
        source: str,
        output_name: str,
        template_context: dict,
):
    """
    Render the yaml-config from source via clipig.py

    :param source: str, either a filename or the yaml text
    :param output_name: complete filename of the output image file
    :param template_context: dict with key/value mapping
        where each instane of `{{key}}` will be replaced with `value`.
    """
    config_text = yaml_template(source, template_context)

    call_args = [
        "python", "clipig.py", "-", "--output", output_name,
        "--snapshot-interval", "20",
                                                ]
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


