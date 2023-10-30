import unittest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from small_batch_linear_model import SmallBatchLinearModel
from small_batch_trainer import SmallBatchTrainer


# noinspection DuplicatedCode
class LargeBatchAndSmallBatchTest(unittest.TestCase):
    def test_train_small_batch(self):
        expected_weight = 2.
        expected_bias = 3.
        learning_rate = 0.01
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_set_x, train_set_y, test_set_x, test_set_y = self.create_data(expected_weight, expected_bias, device)
        model = SmallBatchLinearModel().to(device)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        trainer = SmallBatchTrainer(model, optimizer, loss_function)
        error = trainer.train(train_set_x, train_set_y, batch_size=20, max_epochs=10000, error_threshold=1e-4)
        self.plot_error(error)
        self.evaluate_model(model, test_set_x, test_set_y)

    @staticmethod
    def create_data(expected_weight, expected_bias, device):
        data_set_x = torch.arange(0., 10, 0.01, device=device).unsqueeze(1)
        data_set_y = expected_weight * data_set_x + expected_bias
        train_set_size = int(len(data_set_x) * 0.8)
        train_set_x = data_set_x[:train_set_size]
        train_set_y = data_set_y[:train_set_size]
        test_set_x = data_set_x[train_set_size:]
        test_set_y = data_set_y[train_set_size:]
        return train_set_x, train_set_y, test_set_x, test_set_y

    @staticmethod
    def plot_error(error):
        plt.figure(figsize=(10, 5))
        plt.plot(torch.tensor(error), label='Error')
        plt.legend(prop={'size': 15})
        plt.show()

    @staticmethod
    def evaluate_model(model, test_set_x, test_set_y):
        model.eval()
        print(f'Trained model: {model.state_dict()}')
        with torch.inference_mode():
            test_pred_set_y = model(test_set_x)
            print(f'Loss value on test set: {nn.MSELoss()(test_pred_set_y, test_set_y)}')

        plt.figure(figsize=(10, 5))
        test_set_x_cpu = test_set_x.cpu()
        plt.scatter(test_set_x_cpu, test_set_y.cpu(), label='Expected', c='g')
        plt.scatter(test_set_x_cpu, test_pred_set_y.cpu(), label='Predicted', c='r')
        plt.legend(prop={'size': 15})
        plt.show()


if __name__ == '__main__':
    unittest.main()
