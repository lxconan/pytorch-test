import torch
import torch.nn as nn


def create_linear_data_set(start: float, end: float, step: float, weight: float, bias: float,
                           train_set_factor: float = 0.8, device: torch.device = None):
    x_data_set = torch.arange(start, end, step, device=device).unsqueeze(dim=1)
    y_data_set = weight * x_data_set + bias
    if train_set_factor <= 0 or train_set_factor >= 1:
        raise ValueError('The train_set_factor should be between 0 and 1.')
    train_set_length = int(len(x_data_set) * train_set_factor)
    x_train_set = x_data_set[:train_set_length]
    y_train_set = y_data_set[:train_set_length]
    x_test_set = x_data_set[train_set_length:]
    y_test_set = y_data_set[train_set_length:]
    return x_train_set, y_train_set, x_test_set, y_test_set


class LinearRegressionModelUsingCustomParameters(nn.Module):
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


class DeviceIndependentLinearModel(nn.Module):
    def __init__(self, device):
        super(DeviceIndependentLinearModel, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

