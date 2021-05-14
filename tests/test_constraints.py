import unittest

import torch

from src.expression import ExpressionContext
from src.parameters import EXPR_ARGS
from src import constraints


class TestConstraints(unittest.TestCase):

    def expression_context(self):
        return ExpressionContext(**{
            key: 0
            for key in EXPR_ARGS.TARGET_CONSTRAINT
        })

    def test_noise(self):
        c = constraints.NoiseConstraint([.1, .1, .1], "l2")
        img = torch.rand(3, 10, 11)
        c.forward(img, ExpressionContext())
