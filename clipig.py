import json
import datetime
import pathlib
import time

from src.parameters import parse_arguments, save_yaml_config
from src.files import make_filename_dir, change_extension
from src.training import ImageTraining


if __name__ == "__main__":
    parameters = parse_arguments()
    # print(parameters); exit()
    # print(json.dumps(parameters, indent=2)); exit()

    trainer = ImageTraining(parameters)
    start_time = time.time()

    try:
        trainer.train()

    except KeyboardInterrupt:
        pass

    except RuntimeError as e:
        if str(e).strip():
            print(f"[[[{e}]]]")
            raise

    run_time = time.time() - start_time

    trainer.save_image()

    filename = change_extension(parameters["output"], "yaml")
    if not pathlib.Path(filename).exists():
        header = f"""auto-generated at {datetime.datetime.now()}
runtime: {run_time:.2f} seconds"""

        header = "\n".join(f"# {line}" for line in header.splitlines())

        trainer.log(2, f"exporting config {filename}")
        make_filename_dir(filename)
        save_yaml_config(
            filename, parameters,
            header=header,
        )
