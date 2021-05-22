import unittest

import torch

from src.expression import ExpressionContext

# TODO: for some reason this leads to an import error along
#   the circular import chain between constrains.py and parameters.py
#   but only with the unittest runner
# from src import constraints


class TestConstraints(unittest.TestCase):

    def test_noise(self):
        from src import constraints

        c = constraints.NoiseConstraint([.1, .1, .1], "l2")
        img = torch.rand(3, 10, 11)
        c.forward(img, ExpressionContext())
