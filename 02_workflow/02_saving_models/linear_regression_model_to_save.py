import torch
import torch.nn as nn


class LinearRegressionModelToSave(nn.Module):
    def __init__(self):
        super(LinearRegressionModelToSave, self).__init__()
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x: torch.Tensor):
        return x * self.weight + self.bias
