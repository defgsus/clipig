import unittest

from src.expression import Expression, ExpressionContext


class TestExpression(unittest.TestCase):

    def test_expression(self):
        ctx = ExpressionContext()
        self.assertEqual(1.2, ctx(1.2))
        self.assertEqual(1.2, ctx(Expression(float, "1.2")))
        self.assertEqual([1.2, 3.4], ctx([Expression(float, "1.2"), 3.4]))

    def test_expression_param(self):
        ctx = ExpressionContext(a=2, b=3)
        self.assertEqual(6, ctx(Expression(int, "a * b", "a", "b")))
        self.assertEqual([5, 6], ctx([
            Expression(int, "a+b", "a", "b"),
            Expression(int, "a*b", "a", "b"),
        ]))

    def test_expression_math(self):
        e = Expression(float, "sin(3.14159265)")
        self.assertAlmostEqual(0, e())
