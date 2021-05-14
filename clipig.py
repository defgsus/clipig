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
    epoch_time = run_time / (trainer.epoch + 1)

    trainer.save_image()

    filename = change_extension(parameters["output"], "yaml")
    if not pathlib.Path(filename).exists():
        header = f"""
        auto-generated at {datetime.datetime.now().replace(microsecond=0)}
        epochs: {trainer.epoch+1} / {parameters['epochs']}
        runtime: {run_time:.2f} seconds ({epoch_time:.3f}/epoch)
        """

        header = "\n".join(
            f"# {line.lstrip()}"
            for line in header.splitlines()
            if line.strip()
        )

        trainer.log(2, f"exporting config {filename}")
        make_filename_dir(filename)
        save_yaml_config(
            filename, parameters,
            header=header,
        )
