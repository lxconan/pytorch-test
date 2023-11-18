import unittest
import torch


class TestCalculateGradient(unittest.TestCase):
    def test_should_calculate_gradient(self):
        x = torch.tensor(2., requires_grad=True)
        y = torch.pow(x, 3)
        self.assertTrue(y.requires_grad)

        y.backward()
        self.assertTrue(x.grad.equal(torch.tensor(12.)))  # dy/dx = 3x^2 = 3 * 2^2 = 12

    def test_should_calculate_gradient_for_tensor(self):
        x = torch.tensor([[1., 2., 3], [4., 5., 6.]], requires_grad=True)
        y = torch.matmul(x, torch.tensor([[3.], [2.], [1.]]))

        y.backward(torch.tensor([[1.], [1.]]))
        self.assertTrue(
            x.grad.equal(torch.tensor([[3., 2., 1.], [3., 2., 1.]])))


if __name__ == '__main__':
    unittest.main()
