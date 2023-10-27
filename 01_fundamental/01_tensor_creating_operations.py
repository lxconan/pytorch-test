import unittest
import torch


class TensorCreatingOperationsTest(unittest.TestCase):

    def test_should_be_able_to_create_scalar_tensor(self):
        scalar = torch.tensor(1.0)
        self.assertTrue(scalar.equal(torch.tensor(1.0)))
        self.assertEqual(scalar.ndim, 0)
        self.assertEqual(scalar.shape, torch.Size([]))

    def test_should_be_able_to_create_1_dimensional_tensor_with_1_element(self):
        vector = torch.tensor([1.0])
        self.assertTrue(vector.equal(torch.tensor([1.0])))
        self.assertEqual(vector.ndim, 1)
        self.assertEqual(vector.shape, torch.Size([1]))

    def test_should_be_able_to_create_vector_tensor(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(vector.equal(torch.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(vector.ndim, 1)
        self.assertEqual(vector.shape, torch.Size([3]))

    def test_should_be_able_to_create_matrix_tensor(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        self.assertTrue(matrix.equal(torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])))
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape, torch.Size([2, 3]))