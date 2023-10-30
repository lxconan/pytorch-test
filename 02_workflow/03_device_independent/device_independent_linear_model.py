import torch
import torch.nn as nn


class DeviceIndependentLinearModel(nn.Module):
    def __init__(self, device):
        super(DeviceIndependentLinearModel, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
