import math
import unittest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from device_independent_linear_model import DeviceIndependentLinearModel


class DeviceIndependentRegression(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

    def test_device_independent_regression(self):
        print(f'Device independent regression test running with pyTorch: {torch.__version__}')
        print(f'Running on device: {self.device}')
        x_train_set, y_train_set, x_test_set, y_test_set = self.create_data_set(1.5, 3)
        self.plot_data_set(x_train_set, y_train_set, x_test_set, y_test_set)
        model = DeviceIndependentLinearModel(self.device)
        print(f'Before training, we initialized the model using random values: {model.state_dict()}')
        errors = self.train_model(model, x_train_set, y_train_set, max_epochs=10000)
        self.plot_error(errors)
        print(f'After training, the model parameters are: {model.state_dict()}')
        self.evaluate_model(model, x_train_set, y_train_set, x_test_set, y_test_set)

    # noinspection DuplicatedCode
    @staticmethod
    def create_data_set(expected_weight, expected_bias):
        # by default the data set is created on the cpu, we will move it to the gpu when needed
        full_data_set_x = torch.arange(0., 50., step=0.01).unsqueeze(dim=1)
        full_data_set_y = expected_weight * full_data_set_x + expected_bias
        train_set_size = int(len(full_data_set_x) * 0.8)
        x_train_set = full_data_set_x[:train_set_size]
        y_train_set = full_data_set_y[:train_set_size]
        x_test_set = full_data_set_x[train_set_size:]
        y_test_set = full_data_set_y[train_set_size:]
        return x_train_set, y_train_set, x_test_set, y_test_set

    @staticmethod
    def plot_data_set(x_train_set: torch.Tensor, y_train_set: torch.Tensor, x_test_set: torch.Tensor,
                      y_test_set: torch.Tensor, predictions: torch.Tensor = None):
        plt.figure(figsize=(10, 7))
        plt.title('Linear Regression (Train/Test Sets)')
        plt.scatter(x_train_set, y_train_set, c='blue', label='Train Set', s=2)
        plt.scatter(x_test_set, y_test_set, c='green', label='Test Set', s=2)
        if predictions is not None:
            plt.scatter(x_test_set, predictions, c='red', label='Predictions', s=2)

        plt.legend(prop={'size': 14})
        plt.show()

    def train_model(self, model, x_train_set, y_train_set, max_epochs, error_threshold=1e-4):
        loss_function = nn.MSELoss()

        # When we use the same learning rate but increase the batch size for each epoch, the error accumulation will
        # increase and the model may not converge. For example, please try to set the batch size to 0-50, step=0.01
        # while keep the learning rate as 0.01 to see the effect.
        #
        # If we decrease the learning rate to 0.001, the model will converge again.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        x_train_set_on_device = x_train_set.to(self.device)
        y_train_set_on_device = y_train_set.to(self.device)

        error = []
        for epoch in range(max_epochs):
            model.train()
            y_pred_on_device = model(x_train_set_on_device)
            loss = loss_function(y_pred_on_device, y_train_set_on_device)
            error.append(loss.item())
            if loss.item() < error_threshold:
                print(f'Training stopped after {epoch} epochs, loss: {loss}')
                break
            if math.isnan(loss.item()):
                print(f'Training stopped after {epoch} epochs, because loss value is nan.')
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return error

    @staticmethod
    def plot_error(error):
        plt.figure(figsize=(10, 7))
        plt.title('Training Error')
        plt.plot(torch.arange(0, len(error)), torch.tensor(error), c='blue', label='Error')
        plt.legend(prop={'size': 14})
        plt.show()

    def evaluate_model(self, model, x_train_set, y_train_set, x_test_set, y_test_set):
        x_test_set_on_device = x_test_set.to(self.device)
        y_test_set_on_device = y_test_set.to(self.device)
        model.eval()
        with torch.inference_mode():
            y_pred_on_device = model(x_test_set_on_device)
            self.plot_data_set(x_train_set, y_train_set, x_test_set, y_test_set, predictions=y_pred_on_device.cpu())
            loss = nn.MSELoss()(y_pred_on_device, y_test_set_on_device)
            print(f'Test loss: {loss}')


if __name__ == '__main__':
    unittest.main()
