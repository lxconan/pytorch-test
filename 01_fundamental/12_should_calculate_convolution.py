import unittest
import torch.nn as nn
import torch


class TestCalculateConvolution(unittest.TestCase):
    def test_calculate_convolution_layer(self):
        torch.manual_seed(42)
        images = torch.ones(1, 1, 4, 4)
        conv_layer = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        conv_layer.weight.data = torch.ones(2, 1, 3, 3)
        conv_layer.bias.data = torch.zeros(2)
        output = conv_layer(images)
        self.assertTrue(output.shape == torch.Size([1, 2, 4, 4]))
        self.assertTrue(output.equal(torch.Tensor([[[[4, 6, 6, 4], [6, 9, 9, 6], [6, 9, 9, 6], [4, 6, 6, 4]],
                                                     [[4, 6, 6, 4], [6, 9, 9, 6], [6, 9, 9, 6], [4, 6, 6, 4]]]])))


if __name__ == '__main__':
    unittest.main()
