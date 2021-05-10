import unittest

import torch

from src.expression import ExpressionContext
from src.parameters import EXPR_ARGS
from src import transforms


class TestParameters(unittest.TestCase):

    def assertTensor(self, expected: torch.Tensor, real: torch.Tensor):
        if expected.shape != real.shape:
            raise AssertionError(
                f"Shape mismatch, expected {expected.shape}, got {real.shape}"
                f"\n\nexpected:\n{expected}\n\ngot:\n{real}"
            )
        if not torch.all(expected == real):
            raise AssertionError(
                f"Value mismatch,\nexpected:\n{expected}\n\ngot:\n{real}"
            )

    def expression_context(self):
        return ExpressionContext(**{
            key: 0
            for key in EXPR_ARGS.TARGET_TRANSFORM
        })

    def test_shift(self):
        image = torch.Tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ])

        t = transforms.Shift([1, 0])
        self.assertTensor(
            torch.Tensor([
                [[3, 1, 2], [6, 4, 5], [9, 7, 8]],
                [[13, 11, 12], [16, 14, 15], [19, 17, 18]],
                [[23, 21, 22], [26, 24, 25], [29, 27, 28]],
            ]),
            t(image, self.expression_context())
        )

        t = transforms.Shift([-2, 0])
        self.assertTensor(
            torch.Tensor([
                [[3, 1, 2], [6, 4, 5], [9, 7, 8]],
                [[13, 11, 12], [16, 14, 15], [19, 17, 18]],
                [[23, 21, 22], [26, 24, 25], [29, 27, 28]],
            ]),
            t(image, self.expression_context())
        )

        t = transforms.Shift([-1, 0])
        self.assertTensor(
            torch.Tensor([
                [[2, 3, 1], [5, 6, 4], [8, 9, 7]],
                [[12, 13, 11], [15, 16, 14], [18, 19, 17]],
                [[22, 23, 21], [25, 26, 24], [28, 29, 27]],
            ]),
            t(image, self.expression_context())
        )
