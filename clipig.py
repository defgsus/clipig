import json
import datetime
import pathlib

from src.parameters import parse_arguments, save_yaml_config
from src.files import make_filename_dir, change_extension
from src.training import ImageTraining


if __name__ == "__main__":
    parameters = parse_arguments()
    # print(json.dumps(parameters, indent=2)); exit()

    trainer = ImageTraining(parameters)

    filename = change_extension(parameters["output"], "yaml")
    if not pathlib.Path(filename).exists():
        trainer.log(2, f"exporting config {filename}")
        make_filename_dir(filename)
        save_yaml_config(
            filename, parameters,
            f"# auto-generated at {datetime.datetime.now()}\n"
        )

    try:
        trainer.train()

    except KeyboardInterrupt:
        pass

    except RuntimeError as e:
        if str(e).strip():
            print(f"[[[{e}]]]")
            raise

    trainer.save_image()
