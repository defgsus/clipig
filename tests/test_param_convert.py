import unittest

from src.parameters import set_parameter_defaults, convert_params, merge_parameters


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
        self.assertEqual(.5, params["learnrate"](t=1., epoch=0))
