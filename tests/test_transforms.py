import unittest

import torch

from src.expression import ExpressionContext
from src.parameters import EXPR_GROUPS
from src import transforms


class TestTransforms(unittest.TestCase):

    def assertTensor(self, expected: torch.Tensor, real: torch.Tensor):
        if expected.shape != real.shape:
            raise AssertionError(
                f"Shape mismatch, expected {expected.shape}, got {real.shape}"
                f"\n\nexpected:\n{expected}\n\ngot:\n{real}"
            )
        if not torch.all(torch.abs(expected - real) < 0.0001):
            raise AssertionError(
                f"Value mismatch,\nexpected:\n{expected}\n\ngot:\n{real}"
            )

    def expression_context(self):
        return ExpressionContext(**{
            key: 0
            for key in EXPR_GROUPS.target_transform
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

    def test_pad(self):
        image = torch.Tensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
            [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
        ])

        t = transforms.Pad(size=[1, 0], color=[0, 0, 0], mode="fill")
        self.assertTensor(
            torch.Tensor([
                [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0]],
                [[0, 11, 12, 13, 0], [0, 14, 15, 16, 0], [0, 17, 18, 19, 0]],
                [[0, 21, 22, 23, 0], [0, 24, 25, 26, 0], [0, 27, 28, 29, 0]],
            ]),
            t(image, self.expression_context())
        )

        t = transforms.Pad(size=[1, 0], color=[0, 0, 0], mode="wrap")
        self.assertTensor(
            torch.Tensor([
                [[3, 1, 2, 3, 1], [6, 4, 5, 6, 4], [9, 7, 8, 9, 7]],
                [[13, 11, 12, 13, 11], [16, 14, 15, 16, 14], [19, 17, 18, 19, 17]],
                [[23, 21, 22, 23, 21], [26, 24, 25, 26, 24], [29, 27, 28, 29, 27]],
            ]),
            t(image, self.expression_context())
        )

    def test_quantize(self):
        image = torch.Tensor([
            [[.11, .21, .31], [.41, .51, .61], [.71, .81, .91]],
            [[.13, .23, .33], [.43, .53, .63], [.73, .83, .93]],
            [[.16, .26, .36], [.46, .56, .66], [.76, .86, .96]],
        ])

        t = transforms.Quantize(step=[.1, .3, .5])
        self.assertTensor(
            torch.Tensor([
                [[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
                [[.0, .0, .3], [.3, .3, .6], [.6, .6, .9]],
                [[.0, .0, .0], [.0, .5, .5], [.5, .5, .5]],
            ]),
            t(image, self.expression_context())
        )
