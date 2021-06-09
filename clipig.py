import pathlib
import time
import sys

from src.parameters import parse_arguments, save_yaml_config
from src.files import make_filename_dir, change_extension
from src.training import ImageTraining
from src.files import prepare_output_name


if __name__ == "__main__":

    parameters = parse_arguments()
    num_repeat = parameters.pop("repeat", 1)

    try:

        for i in range(num_repeat):

            trainer = ImageTraining(parameters)
            start_time = time.time()

            trainer.train()

            run_time = time.time() - start_time

            trainer.save_image()

            filename = change_extension(parameters["output"], "yaml")
            if not pathlib.Path(filename).exists():
                trainer.save_yaml(filename, run_time=run_time)

            # update output-name when repeating
            parameters["output"] = str(prepare_output_name(parameters["output"], make_dir=False))

    except KeyboardInterrupt:
        pass

    except RuntimeError as e:
        # sometimes the keyboard interrupt ungracefully breaks
        # the torchscript execution
        if not str(e).strip().endswith("RuntimeError:"):
            raise
