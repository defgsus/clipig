import unittest

import torch

from src import images


class TestPixels(unittest.TestCase):

    def test_resize_crop(self):
        self.assertEqual(
            torch.Size([3, 100, 200]),
            images.resize_crop(torch.randn(3, 300, 300), [100, 200]).shape,
        )
        self.assertEqual(
            torch.Size([3, 100, 200]),
            images.resize_crop(torch.randn(3, 100, 300), [100, 200]).shape,
        )
        self.assertEqual(
            torch.Size([3, 100, 200]),
            images.resize_crop(torch.randn(3, 50, 200), [100, 200]).shape,
        )
        self.assertEqual(
            torch.Size([3, 100, 200]),
            images.resize_crop(torch.randn(3, 300, 50), [100, 200]).shape,
        )
