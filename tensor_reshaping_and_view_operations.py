import unittest
import torch


class TestTensorReshapingAndViewOperations(unittest.TestCase):

    # * Reshaping - reshapes the input tensor to a defined shape
    # * View - returns a view of an input tensor of certain shape but keep the same memory as the original tensor
    # * Stacking - combine multiple tensors on top of each other (vstack) or side by side (hstack)
    # * Squeeze - remove dimensions of size 1 from a tensor
    # * Unsqueeze - add a dimension of size 1 to a tensor
    # * Permute - Returns a view of the input with the dimensions permuted (swapped) in a desired way

    def test_should_reshaping_tensors(self):
        vector_9 = torch.arange(1., 10.)
        self.assertEqual(vector_9.shape, torch.Size([9]))

        # let's add an extra dimension to the vector
        reshaped = vector_9.reshape(3, 3)
        self.assertEqual(reshaped.shape, torch.Size([3, 3]))
        self.assertTrue(reshaped.equal(torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])))

        # we can change to any shape which is compatible with the number of elements
        reshaped = vector_9.reshape(9, 1)
        self.assertEqual(reshaped.shape, torch.Size([9, 1]))
        self.assertTrue(reshaped.equal(torch.tensor([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]])))

    def test_should_raise_error_if_reshaping_is_not_compatible(self):
        vector_9 = torch.arange(1., 10.)
        self.assertRaises(RuntimeError, lambda: vector_9.reshape(3, 4))

    def test_should_change_shape_by_view(self):
        vector_9 = torch.arange(1., 10.)
        view_3_3 = vector_9.view(3, 3)

        self.assertEqual(view_3_3.shape, torch.Size([3, 3]))

    def test_view_shares_memory_with_original_tensor(self):
        vector_9 = torch.arange(1., 10.)
        view_3_3 = vector_9.view(3, 3)
        view_3_3[0, 0] = 100

        self.assertTrue(vector_9.equal(torch.tensor([100., 2., 3., 4., 5., 6., 7., 8., 9.])))

    def test_reshape_will_return_view_if_compatible(self):
        vector_9 = torch.arange(1., 10.)
        matrix_3_3 = vector_9.reshape(3, 3)
        matrix_3_3[0, 0] = 100

        self.assertTrue(vector_9.equal(torch.tensor([100., 2., 3., 4., 5., 6., 7., 8., 9.])))
        self.assertTrue(matrix_3_3.equal(torch.tensor([[100., 2., 3.], [4., 5., 6.], [7., 8., 9.]])))

    def test_should_stack_tensors_on_top_of_each_other_or_side_by_side(self):
        vector_3 = torch.arange(1., 4.)
        self.assertTrue(
            torch.stack([vector_3, vector_3, vector_3]).equal(torch.tensor([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.]])))
        self.assertTrue(
            torch.stack([vector_3, vector_3], dim=1).equal(torch.tensor([[1., 1.], [2., 2.], [3., 3.]])))

    def test_should_squeeze_tensor_by_removing_the_single_dimensions(self):
        original_tensor = torch.arange(1., 4.).reshape(1, 3)
        self.assertEqual(original_tensor.shape, torch.Size([1, 3]))
        self.assertTrue(original_tensor.equal(torch.tensor([[1., 2., 3.]])))

        squeezed = original_tensor.squeeze()
        self.assertEqual(squeezed.shape, torch.Size([3]))
        self.assertTrue(squeezed.equal(torch.tensor([1., 2., 3.])))

    def test_should_unsqueeze_by_adding_single_dimension_to_target_tensor(self):
        original_tensor = torch.arange(1., 4.)
        self.assertEqual(original_tensor.shape, torch.Size([3]))

        unsqueezed_row = original_tensor.unsqueeze(dim=0)
        self.assertEqual(unsqueezed_row.shape, torch.Size([1, 3]))
        self.assertTrue(unsqueezed_row.equal(torch.tensor([[1., 2., 3.]])))

        unsqueezed_column = original_tensor.unsqueeze(dim=1)
        self.assertEqual(unsqueezed_column.shape, torch.Size([3, 1]))
        self.assertTrue(unsqueezed_column.equal(torch.tensor([[1.], [2.], [3.]])))

    def test_should_get_view_with_dimensions_rearranged(self):
        original_tensor = torch.rand(size=(4, 2, 3))  # [height, width, color_channels]
        # shifts axis 0->1, 1->2, 2->0, that will create [color_channels, height, width]
        permuted = original_tensor.permute(2, 0, 1)

        self.assertEqual(permuted.shape, torch.Size([3, 4, 2]))

    def test_the_rearranged_view_can_modify_the_original_value(self):
        original_tensor = torch.rand(size=(4, 2, 3))  # [height, width, color_channels]
        # shifts axis 0->1, 1->2, 2->0, that will create [color_channels, height, width]
        permuted = original_tensor.permute(2, 0, 1)

        permuted[1, 2, 1] = 128

        self.assertEqual(original_tensor[2, 1, 1], 128)



