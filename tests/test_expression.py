import unittest

from src.expression import Expression, ExpressionContext, EXPRESSION_ARGS
from src.parameters import set_parameter_defaults, convert_params


class TestExpression(unittest.TestCase):

    def test_expression(self):
        ctx = ExpressionContext()
        self.assertEqual(1.2, ctx(1.2))
        self.assertEqual(1.2, ctx(Expression(float, "1.2")))
        self.assertEqual([1.2, 3.4], ctx([Expression(float, "1.2"), 3.4]))

    def test_expression_param(self):
        ctx = ExpressionContext(learnrate=2, learnrate_scale=3)
        self.assertEqual(6, ctx(
            Expression(int, "learnrate * learnrate_scale", ("learnrate",))
        ))
        self.assertEqual([5, 6], ctx([
            Expression(int, "learnrate + learnrate_scale", ("learnrate", )),
            Expression(int, "learnrate * learnrate_scale", ("learnrate", )),
        ]))

    def test_expression_math(self):
        e = Expression(float, "sin(3.14159265)")
        self.assertAlmostEqual(0, e())

    def test_eval_wthout_params(self):
        params = {
            "resolution": "224*2 224//2",
        }
        params = convert_params(params)
        set_parameter_defaults(params)
        self.assertEqual([448, 112], params["resolution"])
    
    @unittest.expectedFailure
    def test_lists_as_result_of_expression(self):
        self.assertEqual(
            [1., 2.],
            Expression(float, "[1, 2]")(),
        )
