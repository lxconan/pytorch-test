import unittest
import math

import torch
import torch.nn as nn

import common_features.linear as linear
import common_features.plot as plt


# noinspection DuplicatedCode
class LargeBatchAndSmallBatchTest(unittest.TestCase):
    def test_train_small_batch(self):
        expected_weight = 2.
        expected_bias = 3.
        learning_rate = 0.01
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_set_x, train_set_y, test_set_x, test_set_y = linear.create_linear_data_set(
            0., 10, 0.01, expected_weight, expected_bias, 0.8, device)
        model = linear.DeviceIndependentLinearModel(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        error = self.train(
            model, optimizer, train_set_x, train_set_y, batch_size=20, max_epochs=10000, error_threshold=1e-4)
        plt.plot_loss_values(torch.Tensor(error))
        self.evaluate_model(model, test_set_x, test_set_y)

    @staticmethod
    def evaluate_model(model, test_set_x, test_set_y):
        model.eval()
        print(f'Trained model: {model.state_dict()}')
        with torch.inference_mode():
            test_pred_set_y = model(test_set_x)
            print(f'Loss value on test set: {nn.MSELoss()(test_pred_set_y, test_set_y)}')

        plt.plot_linear_training_set_and_expected_test_set(test_set_x, test_set_y, test_set_x, test_pred_set_y)

    @staticmethod
    def train(model, optimizer, train_set_x, train_set_y, batch_size: int, max_epochs: int,
              error_threshold: float):
        errors = []
        loss_function = nn.MSELoss()
        for epoch in range(max_epochs):
            model.train()
            for batch_train_set in zip(train_set_x.tensor_split(batch_size), train_set_y.tensor_split(batch_size)):
                batch_train_set_x = batch_train_set[0]
                batch_train_set_y = batch_train_set[1]
                batch_pred_set_y = model(batch_train_set_x)
                batch_loss = loss_function(batch_pred_set_y, batch_train_set_y)
                errors.append(batch_loss)
                if batch_loss < error_threshold:
                    print(f'Training stopped at epoch "{epoch}" with error "{batch_loss}"')
                    return errors
                if math.isnan(batch_loss):
                    print(f'Training stopped at epoch "{epoch}" because the error became "nan"')
                    return errors
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        return errors


if __name__ == '__main__':
    unittest.main()
