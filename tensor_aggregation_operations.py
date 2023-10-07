from unittest import TestCase

import torch


class TestTensorAggregationOperations(TestCase):

    def test_should_be_able_to_get_max_value_for_0_dim_tensor(self):
        scalar = torch.tensor(1, dtype=torch.int16)
        self.assertEqual(scalar.max().item(), 1)

    def test_should_be_able_to_get_max_value_for_1_dim_tensor(self):
        vector = torch.tensor([1, 2, 3], dtype=torch.int16)
        self.assertEqual(vector.max().item(), 3)

    def test_should_be_able_to_get_max_value_for_2_dim_tensor(self):
        matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16)
        self.assertEqual(matrix.max().item(), 6)

    def test_should_be_able_to_get_max_value_for_multiple_dim_tensor(self):
        tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6]],
             [[7, 8, 9], [0, 11, 12]]], dtype=torch.int16)
        self.assertEqual(tensor.max().item(), 12)

    def test_should_be_able_to_get_mean_value_for_1_dim_tensor(self):
        # Please note that the mean value only supports float and complex type.
        vector = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        self.assertEqual(vector.mean().item(), 2.5)

    def test_should_be_able_to_get_mean_value_for_2_dim_tensor(self):
        # Just flatten as one dim vector and the aggregate.
        matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        self.assertEqual(matrix.mean().item(), 3.5)

    def test_should_convert_to_float_point_first_before_mean_op(self):
        int_matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16)
        # Mean only supports float and complex, so we can convert type first
        float_matrix = int_matrix.type(torch.float32)
        self.assertEqual(torch.mean(float_matrix).item(), 3.5)

    def test_should_be_able_to_get_sum_of_tensor(self):
        vector = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        self.assertEqual(vector.sum().item(), 10)

    def test_should_be_able_to_get_sum_of_multiple_dims_tensor(self):
        tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6]],
             [[7, 8, 9], [0, 11, 12]]], dtype=torch.int16)
        self.assertEqual(tensor.sum().item(), 68)

    def test_should_return_the_index_of_the_min_max_value(self):
        vector = torch.tensor([100, 200, 300, 400], dtype=torch.int16)
        self.assertEqual(vector.argmax().item(), 3)
        self.assertEqual(vector.argmin().item(), 0)

    def test_should_return_the_index_of_the_min_value_for_multiple_dims_tensor(self):
        tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6]],
             [[7, 8, 9], [0, 11, 12]]], dtype=torch.int16)
        self.assertEqual(tensor.argmin().item(), 9)
        self.assertEqual(tensor.argmax().item(), 11)

    