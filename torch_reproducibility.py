import unittest
import torch


class TestTorchReproducibility(unittest.TestCase):

    # Reproducibility - the ability to reproduce the same results given the same input.
    # This is important for neural networks because when neural network learns, it will do the following:
    #
    # * Start with random numbers
    # * Doing tensor operations
    # * update random numbers to try and make them better representations of the data
    # * Again, and again ...
    #
    # If the random numbers are not reproducible, then the results will not be reproducible. To reduce the randomness
    # in PyTorch, we can set the seed for the random number generator.
    #
    # For pytorch external resource of reproducibility, please refer to:
    # * https://pytorch.org/docs/stable/notes/randomness.html
    # * https://en.wikipedia.org/wiki/Randomness

    def test_should_create_2_random_tensors(self):
        random_tensor_1 = torch.rand(2, 2)
        random_tensor_2 = torch.rand(2, 2)

        self.assertFalse(random_tensor_1.equal(random_tensor_2))

    def test_should_create_2_same_equal_random_tensors(self):
        random_seed = 100
        torch.manual_seed(random_seed)  # The manual_seed function will only service one random number generation.
        random_tensor_1 = torch.rand(2, 2)

        torch.manual_seed(random_seed)
        random_tensor_2 = torch.rand(2, 2)

        self.assertTrue(random_tensor_1.equal(random_tensor_2))
