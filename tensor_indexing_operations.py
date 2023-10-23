import unittest

import torch


class TestTensorIndexingOperations(unittest.TestCase):

    def test_should_selecting_tensor_from_tensors_by_first_dimension(self):
        tensor = torch.arange(1., 10.).reshape(1, 3, 3)
        # [[[1., 2., 3.],
        #   [4., 5., 6.],
        #   [7., 8., 9.]]]

        self.assertTrue(tensor[0].equal(torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])))

    def test_should_selecting_tensor_from_tensor_by_second_dimension(self):
        tensor = torch.arange(1., 10.).reshape(1, 3, 3)
        # [[[1., 2., 3.],
        #   [4., 5., 6.],
        #   [7., 8., 9.]]]

        # In PyTorch, there are differences between the two indexing operations: tensor[1][2] and tensor[1, 2]
        #
        # For `tensor[1][2]`, this is a chained indexing operation. The first [1] indexes the tensor to access a
        # sub-tensor (a row) at index 1. The second [2] indexes the sub-tensor obtained in the previous step to access
        # a specific element within that sub-tensor. Essentially, it first extracts a row and then accesses an element
        # within that row.
        #
        # For `tensor[1, 2]`, this is a single indexing operation. The [1, 2] indexes the tensor to access a specific
        # element at index 1, 2. Essentially, it directly accesses an element at a specific index.
        #
        # In terms of preference, tensor[1, 2] is generally preferred for several reasons:
        # 1. It is more concise and easier to read.
        # 2. It is more efficient as it only performs one indexing operation.
        # 3. It is consistent with the indexing operation for NumPy arrays, and with standard Python indexing.
        self.assertTrue(tensor[0][1].equal(torch.tensor([4., 5., 6.])))
        self.assertTrue(tensor[0, 1].equal(torch.tensor([4., 5., 6.])))

    def test_should_selecting_tensor_from_tensor_by_most_inner_dimension(self):
        tensor = torch.arange(1., 10.).reshape(1, 3, 3)
        # [[[1., 2., 3.],
        #   [4., 5., 6.],
        #   [7., 8., 9.]]]

        self.assertTrue(tensor[0, 1, 2] == torch.tensor(6.))

    def test_should_raise_error_if_index_is_out_of_bounds(self):
        tensor = torch.arange(1., 10.).reshape(1, 3, 3)
        # [[[1., 2., 3.],
        #   [4., 5., 6.],
        #   [7., 8., 9.]]]

        self.assertRaises(IndexError, lambda: tensor[1, 0, 0])

    def test_should_select_all_from_target_dimension(self):
        tensor = torch.arange(1., 10.).reshape(1, 3, 3)
        # [[[1., 2., 3.],
        #   [4., 5., 6.],
        #   [7., 8., 9.]]]

        # get all values from the first elements in the first dimension and the second value in the second dimension
        # but all values from the third dimension
        self.assertTrue(tensor[0, 1, :].equal(torch.tensor([4., 5., 6.])))

        # get all values from the first dimension and the second dimension, but only get the third value from the last
        # dimension
        self.assertTrue(tensor[:, :, 2].shape == torch.Size([1, 3]))
        self.assertTrue(tensor[:, :, 2].equal(torch.tensor([[3., 6., 9.]])))

        # get all values from the first dimension and the first value in the second dimension and the first value in
        # the last dimension. Note that we should distinguish the difference between tensor[:, 0, 0] and tensor[0, 0, 0]
        self.assertTrue(tensor[:, 0, 0].equal(torch.tensor([1.])))
