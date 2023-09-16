import torch
import unittest


class TensorGettingOperationsTest(unittest.TestCase):

    def test_should_be_able_to_get_item_for_zero_dim_tensor(self):
        scalar = torch.tensor(1.0)
        self.assertEqual(scalar.item(), 1.0)

    def test_should_be_able_to_get_item_for_1_dim_tensor(self):
        vector = torch.tensor([1.0])
        self.assertEqual(vector.item(), 1.0)

    def test_should_be_able_to_get_item_for_2_dim_tensor(self):
        vector = torch.tensor([[2.0]])
        self.assertEqual(vector.item(), 2.0)

    def test_should_be_able_to_get_item_for_more_dim_tensor(self):
        vector = torch.tensor([[[[3.0]]]])
        self.assertEqual(vector.item(), 3.0)
