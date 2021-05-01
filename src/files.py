import glob
import os
import pathlib
import re


def prepare_output_name(output: str, make_dir: bool = False) -> str:
    """
    Fully resolve the output name and attach a new number
    when a file already exists.

    :param output: str, filename or path with filename
    :param make_dir: bool, create directory if not present
    :return: str, fully path-expanded and unique filename
    """
    assert not output.endswith(os.path.sep), "filename required"

    filepath = pathlib.Path(output).resolve()

    if filepath.exists():

        filename = filepath.name
        extension = None
        if "." in filename:
            filename = filename.split(".")
            extension = filename.pop(-1)
            filename = ".".join(filename)

        existing_names = list(glob.glob(
            str(filepath.parent.joinpath(filename)) + "*"
        ))
        number_re = re.compile(r'^-\d+\.')
        existing_numbers = list(filter(bool, [
             number_re.findall(pathlib.Path(fn).name[len(filename):]) for fn in existing_names
        ]))

        if existing_numbers:
            max_number = max(map(
                lambda f: int(f[0][1:-1]),
                existing_numbers
            ))
        else:
            max_number = 0

        filename = f"{filename}-{max_number+1}"
        if extension:
            filename += f".{extension}"

        filepath = filepath.parent.joinpath(filename)

        assert not filepath.exists(), (
            f"Something went wrong generating a new filename for "
            f"'{output}', got existing '{filepath}'"
        )

    if make_dir and not filepath.parent.exists():
        os.makedirs(filepath.parent)

    return filepath


def make_filename_dir(filename: str):
    filepath = pathlib.Path(filename)
    if not filepath.parent.exists():
        os.makedirs(filepath.parent)
