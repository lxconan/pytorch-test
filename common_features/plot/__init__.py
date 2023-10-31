import torch
import matplotlib.pyplot as plt


def plot_linear_training_set_and_expected_test_set(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor,
                                                   test_y: torch.Tensor, predictions: torch.Tensor = None):
    plt.figure(figsize=(10, 7))
    plt.title(('Linear Regression (Train/Test Sets)' if predictions is None
               else 'Linear Regression (Train/Test/Predictions Sets)'))
    train_x = train_x.cpu() if train_x.is_cuda else train_x
    train_y = train_y.cpu() if train_y.is_cuda else train_y
    test_x = test_x.cpu() if test_x.is_cuda else test_x
    test_y = test_y.cpu() if test_y.is_cuda else test_y
    plt.scatter(train_x, train_y, c='blue', label='Train Set', s=2)
    plt.scatter(test_x, test_y, c='green', label='Test Set', s=2)
    if predictions is not None:
        predictions = predictions.cpu() if predictions.is_cuda else predictions
        plt.scatter(test_x, predictions, c='red', label='Predictions', s=2)

    plt.legend(prop={'size': 14})
    plt.show()


def plot_loss_values(loss_values: torch.Tensor):
    plt.figure(figsize=(10, 7))
    plt.title('Training Error')
    plt.plot(torch.arange(0, len(loss_values), step=1), loss_values, label='Training loss')
    plt.legend(prop={'size': 14})
    plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
