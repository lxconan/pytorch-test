import unittest

import torch


class TestTensorMatrixRelatedOperations(unittest.TestCase):

    def test_should_get_transpose(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.assertTrue(tensor.T.equal(torch.tensor([[1, 4], [2, 5], [3, 6]])))

    def test_should_get_transpose_for_multiple_dims_matrix(self):
        # plane-0                    # plane-0
        # [1, 2, 3]                  # [1, 4]
        # [4, 5, 6] -- transpose --> # [2, 5]
        # plane-1                    # [3, 6]
        # [7, 8, 9]                  # plane-1
        # [10, 11, 12]               # [7, 10]
        #                            # [8, 11]
        #                            # [9, 12]
        # Keeping the plane while transposing the matrix for each plane

        tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        print(tensor.mT)
