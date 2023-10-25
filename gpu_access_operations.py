import unittest

import numpy
import torch


class TestGpuAccessOperations(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("CUDA is mandatory for this test. Please check your environment.")

    def test_should_test_if_gpu_is_supported(self):
        # This test is not for assertion but print useful information on your local GPU information
        # Please check the following document: https://pytorch.org/docs/stable/notes/cuda.html#best-practices
        print("Is CUDA available: ", torch.cuda.is_available())
        print("CUDA device count: ", torch.cuda.device_count())
        print("CUDA device name: ", torch.cuda.get_device_name(0))
        print("CUDA current device: ", torch.cuda.current_device())
        print("CUDA device capability: ", torch.cuda.get_device_capability(0))
        print("CUDA device memory: ", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, "GB")

    def test_should_put_tensors_from_cpu_to_gpu(self):
        tensor_on_cpu = torch.arange(1., 8.)
        self.assertFalse(tensor_on_cpu.is_cuda)

        tensor_on_gpu = tensor_on_cpu.to('cuda')
        self.assertTrue(tensor_on_gpu.is_cuda)

    def test_should_create_tensors_on_gpu(self):
        tensor = torch.arange(1., 8., device='cuda')
        self.assertTrue(tensor.is_cuda)
        self.assertEqual(tensor.device, torch.device('cuda', 0))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(tensor.equal(torch.tensor([1., 2., 3., 4., 5., 6., 7.], device='cuda')))

    def test_should_move_tensors_back_to_cpu(self):
        # For example, some of the commonly used library such as NumPy does not support GPU.
        # Therefore, we need to move the tensor back to CPU.
        tensor_on_gpu = torch.arange(1., 8., device='cuda')
        self.assertTrue(tensor_on_gpu.is_cuda)

        # Re-assign the tensor to cpu and then converted to numpy array
        array_on_cpu = tensor_on_gpu.cpu().numpy()
        self.assertTrue(numpy.array_equal(array_on_cpu, numpy.arange(1., 8.)))