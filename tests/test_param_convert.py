import unittest
from itertools import chain

from src.parameters import set_parameter_defaults, convert_params, merge_parameters, EXPR_GROUPS
from src.expression import EXPRESSION_ARGS


class TestParametersConvert(unittest.TestCase):

    def test_no_expression(self):
        params = {
            "learnrate": 1.,
        }
        params = convert_params(params)
        set_parameter_defaults(params)
        self.assertEqual(float, type(params["learnrate"]))
        self.assertEqual(1.0, params["learnrate"])

    def test_expression(self):
        params = {
            "learnrate": "t * .5",
        }
        params = convert_params(params)
        set_parameter_defaults(params)
        args = {
            key: 0.
            for key in chain(EXPRESSION_ARGS["basic"]["args"], EXPRESSION_ARGS["resolution"]["args"])
        }
        args["time"] = .5
        self.assertEqual(.25, params["learnrate"](**args))
