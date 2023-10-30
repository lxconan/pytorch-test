import torch
import torch.nn as nn


class SmallBatchLinearModel(nn.Module):
    def __init__(self):
        super(SmallBatchLinearModel, self).__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    