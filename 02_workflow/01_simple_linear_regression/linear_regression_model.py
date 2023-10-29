import torch
from torch import nn


class LinearRegressionModel(nn.Module):
    # When we would like to build our own module, we need to derived from the nn.Module class.
    def __init__(self):
        super().__init__()

        # When we are using neural network to do linear regression prediction using gradient descent, we need to
        # initialize random weights and bias. We will use pytorch's nn.Parameter to create the weights and bias.
        # Since we would like to use gradient descent to update the weights and bias, we need to set the
        # `requires_grad` to True. Parameters are a special kind of Tensor. The reason we use `Parameter` is that
        # the parameter will auto bind itself to the module. And then we can use `parameters()` method to get all the
        # parameters of the module.
        #
        # More accurately, when we set `requires_grad` to True, the pytorch will track the gradient of this specific
        # parameter for use with `torch.autograd` (which implements gradient decent).
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Any subclass of `nn.Module` needs to override `forward()` method. This method defines the forward computation
        # for the module. In our case, it is the linear regression formular: y = wx + b.
        return self.weights * x + self.bias