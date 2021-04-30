import json

from src.parameters import parse_arguments


if __name__ == "__main__":
    parameters = parse_arguments()

    print(json.dumps(parameters, indent=2))
