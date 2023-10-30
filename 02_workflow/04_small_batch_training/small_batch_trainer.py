import math

import torch.nn as nn


class SmallBatchTrainer:
    def __init__(self, model: nn.Module, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, train_set_x, train_set_y, batch_size: int, max_epochs: int, error_threshold: float):
        errors = []
        for epoch in range(max_epochs):
            self.model.train()
            for batch_train_set in zip(train_set_x.tensor_split(batch_size), train_set_y.tensor_split(batch_size)):
                batch_train_set_x = batch_train_set[0]
                batch_train_set_y = batch_train_set[1]
                batch_pred_set_y = self.model(batch_train_set_x)
                batch_loss = self.loss_function(batch_pred_set_y, batch_train_set_y)
                errors.append(batch_loss)
                if batch_loss < error_threshold:
                    print(f'Training stopped at epoch "{epoch}" with error "{batch_loss}"')
                    return errors
                if math.isnan(batch_loss):
                    print(f'Training stopped at epoch "{epoch}" because the error became "nan"')
                    return errors
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

        return errors
