import torch
import unittest


class TensorMatrixMultiplicationTest(unittest.TestCase):

    def test_should_not_matrix_multiply_tensors_for_0_dims(self):
        scalar1 = torch.tensor(1.0)
        scalar2 = torch.tensor(2.0)
        self.assertRaises(RuntimeError, lambda: torch.matmul(scalar1, scalar2))

    def test_should_matrix_multiply_tensors_for_1_dims_with_1_element_and_returns_scalar(self):
        vector1 = torch.tensor([2], dtype=torch.int16)
        vector2 = torch.tensor([3], dtype=torch.int16)
        result = torch.matmul(vector1, vector2)
        self.assertFalse(result.equal(torch.tensor([6], dtype=torch.int16)))
        self.assertTrue(result.equal(torch.tensor(6, dtype=torch.int16)))
        self.assertEqual(result.ndim, 0)
        self.assertEqual(result.shape, torch.Size([]))

    def test_should_return_scalar_for_1_dimensional_vector_product_as_dot_product(self):
        vector1 = torch.tensor([1, 2], dtype=torch.int16)
        vector2 = torch.tensor([3, 4], dtype=torch.int16)

        # will be
        # 1 2 dot product 3 4 = 1*3 + 2*4 = 11

        result = torch.matmul(vector1, vector2)
        self.assertTrue(result.equal(torch.tensor(11, dtype=torch.int16)))
        self.assertEqual(result.ndim, 0)

    def test_should_return_1_dimensional_tensor_for_1_dimensional_matrix_product(self):
        vector1 = torch.tensor([2, 3], dtype=torch.int16)
        vector2 = torch.tensor([[3], [4]], dtype=torch.int16)

        # will be
        # 2 3 @ 3 = 18
        #       4

        result = torch.matmul(vector1, vector2)
        self.assertTrue(result.equal(torch.tensor([18], dtype=torch.int16)))

    def test_should_prepend_1s_for_1_dimensional_matrix_multiply_2_dimensional(self):
        one_dimensional = torch.tensor([2, 3], dtype=torch.int16)
        two_dimensional = torch.tensor([[3, 4], [5, 6]], dtype=torch.int16)

        # will be
        # 2 3 @ 3 4 = 21 26
        # 1 1   5 6   8  10
        #
        # but the prepended dimension will be removed.

        result = torch.matmul(one_dimensional, two_dimensional)
        self.assertTrue(result.equal(torch.tensor([21, 26], dtype=torch.int16)))

    def test_should_prepend_1s_for_1_dimensional_matrix_multiply_2_dimensional_non_square_matrix(self):
        one_dimensional = torch.tensor([2, 3], dtype=torch.int16)
        two_dimensional = torch.tensor([[3, 4, 5], [5, 6, 7]], dtype=torch.int16)

        # will be
        # 2 3 @ 3 4 5 = 21 26 31
        # 1 1   5 6 7   8  10 12
        #
        # but the prepended dimension will be removed.

        result = torch.matmul(one_dimensional, two_dimensional)
        self.assertTrue(result.equal(torch.tensor([21, 26, 31], dtype=torch.int16)))

    def test_should_prepend_1s_for_1_dimensional_matrix_multiply_2_dimensional_non_square_thin_matrix(self):
        one_dimensional = torch.tensor([2, 3, 4], dtype=torch.int16)
        two_dimensional = torch.tensor([[3, 4], [5, 6], [7, 8]], dtype=torch.int16)

        # will be
        # 2 3 4 @ 3 4 = 49 58
        # 1 1 1   5 6   15 18
        #         7 8
        # but the prepended dimension will be removed.

        result = torch.matmul(one_dimensional, two_dimensional)
        self.assertTrue(result.equal(torch.tensor([49, 58], dtype=torch.int16)))

    def test_should_return_matrix_vector_product_for_2_dim_and_1_dim_matrix_multiply(self):
        matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.int16)
        vector = torch.tensor([5, 6], dtype=torch.int16)

        # will be
        # 1 2 @ 5 = 17
        # 3 4   6   39

        result = torch.matmul(matrix, vector)
        self.assertTrue(result.equal(torch.tensor([17, 39], dtype=torch.int16)))

    def test_should_return_batched_matrix_multiply_for_3_dim_argument(self):
        batched_matrix = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.int16)
        vector = torch.tensor([[4, 3], [2, 1]], dtype=torch.int16)

        # will be
        # plane-1
        # 1 2 @ 4 3 =  8  5
        # 3 4   2 1   20 13
        # plane-2
        # 5 6 @ 4 3 = 32 21
        # 7 8   2 1   44 29

        result = torch.matmul(batched_matrix, vector)
        self.assertTrue(result.equal(torch.tensor([[[8, 5], [20, 13]], [[32, 21], [44, 29]]], dtype=torch.int16)))

    def test_should_return_batched_matrix_multiply_for_1_dim_vector_as_first_argument(self):
        vector = torch.tensor([1, 2])
        matrix = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

        # will be
        # plane-1
        # 1 2 @ 1 2 =   7  10
        #       3 4     4   6 (should be ignored)
        # plane-2
        # 1 2 @ 5 6 =  19  22
        #       7 8    12  14 (should be ignored)

        result = torch.matmul(vector, matrix)
        self.assertTrue(result.equal(torch.tensor([[7, 10], [19, 22]])))

    def test_should_return_batched_matrix_multiply_for_1_dim_vector_as_second_argument(self):
        matrix = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        vector = torch.tensor([1, 2])

        # will be
        # plane-1
        # 1 2 @ 1 =  5 3
        # 3 4   2   11 7
        # plane-2
        # 5 6 @ 1 = 17 11
        # 7 8   2   23 15

        result = torch.matmul(matrix, vector)
        self.assertTrue(result.equal(torch.tensor([[5, 11], [17, 23]])))

    # TODO:
    # Only broadcast cases are missing now.
    # * If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a
    # batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for
    # the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is
    # appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix
    # (i.e. batch) dimensions are broadcast (and thus must be broadcast-able). For example, if `input` is a
    # (j * 1 * n * n) tensor and `other` is a (k * n * n) tensor, out will be an (j * k * n * n) tensor.
    # Please refer to: https://pytorch.org/docs/stable/generated/torch.matmul.html
