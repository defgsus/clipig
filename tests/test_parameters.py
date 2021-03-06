import unittest

from src import transforms, constraints
from src.parameters import (
    yaml_to_parameters, set_parameter_defaults, convert_params, merge_parameters,
    PARAMETERS,
)
from src.expression import Expression, ExpressionContext, EXPRESSION_ARGS


class TestParameters(unittest.TestCase):

    def get_expression_context(self, *groups: str) -> ExpressionContext:
        params = dict()
        for group in groups:
            for name in EXPRESSION_ARGS[group]["args"]:
                params[name] = 0
        return ExpressionContext(**params)

    def test_set_default_params_empty(self):
        params = dict()
        set_parameter_defaults(params)
        self.assertEqual(1.0, params["learnrate"])
        self.assertEqual([.5, .5, .5], params["init"]["mean"])
        self.assertEqual([], params["targets"])

    def test_set_default_params_dict(self):
        params = {
            "init": {"mean": 23}
        }
        set_parameter_defaults(params)
        self.assertEqual(1.0, params["learnrate"])
        self.assertEqual(23, params["init"]["mean"])
        self.assertEqual([.1, .1, .1], params["init"]["std"])
        self.assertEqual([], params["targets"])

    def test_set_default_params_list_dict(self):
        params = {
            "targets": [
                {"weight": 23},
                {
                    "weight": 42,
                    "constraints": [
                        {"saturation": {"above": 10.}},
                    ]
                },
            ],
        }
        set_parameter_defaults(params)
        self.assertEqual(1.0, params["learnrate"])
        self.assertEqual(23, params["targets"][0]["weight"])
        self.assertEqual(42, params["targets"][1]["weight"])
        self.assertEqual([], params["targets"][0]["features"])
        self.assertEqual(10., params["targets"][1]["constraints"][0]["saturation"]["above"])
        self.assertEqual(1., params["targets"][1]["constraints"][0]["saturation"]["weight"])

    def test_conversion(self):
        params = {
            "resolution": "23",
            "init": {
                "mean": "1",
                "std": "1 2 3",
            },
            "targets": [
                {
                    "start": 1,
                    "end": 0.5,
                    "weight": 23,
                },
                {
                    "start": "25%",
                    "end": "300",
                },
                {
                    "start": "0.5",
                    "end": "99.9%",
                    "transforms": [
                        {"resize": [22, 23]}
                    ]
                },
            ],
        }
        params = convert_params(params)
        set_parameter_defaults(params)
        self.assertEqual([23, 23], params["resolution"])
        self.assertEqual([1, 1, 1], params["init"]["mean"])
        self.assertEqual([1, 2, 3], params["init"]["std"])

        self.assertEqual(float, type(params["targets"][0]["weight"]))
        self.assertEqual(23., params["targets"][0]["weight"])

        self.assertEqual(int, type(params["targets"][0]["start"]))
        self.assertEqual(1, params["targets"][0]["start"])
        self.assertEqual(float, type(params["targets"][0]["end"]))
        self.assertEqual(0.5, params["targets"][0]["end"])

        self.assertEqual(float, type(params["targets"][1]["start"]))
        self.assertEqual(0.25, params["targets"][1]["start"])
        self.assertEqual(int, type(params["targets"][1]["end"]))
        self.assertEqual(300, params["targets"][1]["end"])

        self.assertEqual(float, type(params["targets"][2]["start"]))
        self.assertEqual(0.5, params["targets"][2]["start"])
        self.assertEqual(float, type(params["targets"][2]["end"]))
        self.assertAlmostEqual(0.999, params["targets"][2]["end"])

        self.assertEqual([22, 23], params["targets"][2]["transforms"][0]["resize"])

    def test_merge(self):
        params1 = {
            "resolution": 32,
            "init": {"mean": 2, "std": 3},
            "targets": [
                {"weight": 23}
            ]
        }
        params2 = {
            "resolution": 33,
            "init": {"std": 5},
            "targets": [
                {"weight": 42}
            ]
        }
        params = merge_parameters(params1, params2)
        self.assertEqual(33, params["resolution"])
        self.assertEqual(2, params["init"]["mean"])
        self.assertEqual(5, params["init"]["std"])
        self.assertEqual(2, len(params["targets"]))

    def test_lists_with_multiple_lengths(self):
        yaml = """
        targets:
          - transforms:
              - random_shift: 3
        """
        params = convert_params(yaml_to_parameters(yaml))
        self.assertEqual(
            [3., 3.],
            params["targets"][0]["transforms"][0]["random_shift"],
        )

        yaml = """
        targets:
          - transforms:
              - random_shift: 2 3
        """
        params = convert_params(yaml_to_parameters(yaml))
        self.assertEqual(
            [2., 3.],
            params["targets"][0]["transforms"][0]["random_shift"],
        )

        yaml = """
        targets:
          - transforms:
              - random_shift: 1 2 3
        """
        with self.assertRaises(ValueError):
            params = convert_params(yaml_to_parameters(yaml))

        yaml = """
        targets:
          - transforms:
              - random_shift: 1 2 3 4
        """
        params = convert_params(yaml_to_parameters(yaml))
        self.assertEqual(
            [1., 2., 3., 4.],
            params["targets"][0]["transforms"][0]["random_shift"],
        )

    @unittest.expectedFailure
    def test_lists_as_result_of_expressions(self):
        context = self.get_expression_context(
            *PARAMETERS["targets.transforms.random_shift"].expression_groups,
        )
        yaml = """
        targets:
          - transforms:
              - random_shift:
                - [1, 2]
        """
        params = convert_params(yaml_to_parameters(yaml))

        self.assertEqual(
            [1., 2.],
            context(params["targets"][0]["transforms"][0]["random_shift"]),
        )
