import json

from src.parameters import parse_arguments
from src.training import ImageTraining


if __name__ == "__main__":
    parameters = parse_arguments()
    # print(json.dumps(parameters, indent=2)); exit()

    trainer = ImageTraining(parameters)

    try:
        trainer.train()

    except KeyboardInterrupt:
        pass

    except RuntimeError as e:
        if str(e).strip():
            raise

