import torch
import torch.nn as nn

from linear_regression_model_to_save import LinearRegressionModelToSave


class LinearRegressionModelTrainer:
    def __init__(self, model: LinearRegressionModelToSave):
        self.model = model

    def train(self, max_epochs: int, min_error: float, train_set_x: torch.Tensor, train_set_y: torch.Tensor):
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        for epoch in range(max_epochs):
            self.model.train()
            pred_y = self.model(train_set_x)
            loss = loss_function(pred_y, train_set_y)
            if loss < min_error:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

