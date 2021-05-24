import json
import datetime
import pathlib
import time
import sys

from src.parameters import parse_arguments, save_yaml_config
from src.files import make_filename_dir, change_extension
from src.training import ImageTraining


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "render-documentation":
        from src.doc import render_documentation
        render_documentation()
        exit()

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
        # sometimes the keyboard interrupt ungracefully breaks
        # the torchscript execution
        if not str(e).strip().endswith("RuntimeError:"):
            raise

    run_time = time.time() - start_time

    trainer.save_image()

    filename = change_extension(parameters["output"], "yaml")
    if not pathlib.Path(filename).exists():
        trainer.save_yaml(filename, run_time=run_time)
