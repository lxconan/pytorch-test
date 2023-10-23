import unittest
import torch
import numpy as np


class TestTensorWorkWithNumPy(unittest.TestCase):

    def test_should_put_data_from_numpy_to_tensor(self):
        array = np.arange(1.0, 8.0)
        tensor = torch.from_numpy(array)

        self.assertTrue(tensor.equal(torch.tensor([1., 2., 3., 4., 5., 6., 7.])))
        # please note that the default data type in numpy is float64, while the default data type in PyTorch is float32
        self.assertTrue(tensor.dtype == torch.float64)

        tensor32 = torch.from_numpy(array).type(torch.float32)
        self.assertTrue(tensor32.equal(torch.tensor([1., 2., 3., 4., 5., 6., 7.])))
        self.assertTrue(tensor32.dtype == torch.float32)

    def test_will_share_memory_to_tensor_when_array_element_changes(self):
        array = np.arange(1.0, 8.0)
        tensor = torch.from_numpy(array)

        array[0] = 100
        self.assertTrue(tensor.equal(torch.tensor([100., 2., 3., 4., 5., 6., 7.])))

    def test_will_disconnect_with_new_array_when_array_reference_are_changed(self):
        array = np.arange(1.0, 8.0)
        tensor = torch.from_numpy(array)

        array = array + 1
        self.assertTrue(np.array_equal(array, np.arange(2.0, 9.0)))
        self.assertTrue(tensor.equal(torch.tensor([1., 2., 3., 4., 5., 6., 7.])))

    def test_should_put_data_from_tensor_to_numpy(self):
        tensor = torch.arange(1., 8.)
        array = tensor.numpy()

        self.assertTrue(np.array_equal(array, np.arange(1., 8.)))
        self.assertTrue(array.dtype == np.float32)

    def test_should_share_memory_between_tensor_and_numpy_when_put_data_from_tensor_to_numpy(self):
        tensor = torch.arange(1., 8.)
        array = tensor.numpy()

        tensor[0] = 100
        self.assertEqual(array[0], 100)

    def test_should_convert_to_and_from_numpy_array_to_cpu_first_and_then_they_are_disconnected(self):
        tensor = torch.arange(1., 8., device='cuda')
        array = tensor.cpu().numpy()

        tensor[0] = 100
        self.assertEqual(array[0], 1)
