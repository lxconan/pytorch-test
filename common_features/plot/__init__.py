import torch
import matplotlib.pyplot as plt
import numpy as np


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


def plot_decision_boundary(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    x, y = x.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    x_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(x_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # multi-class
    else:
        y_pred = torch.round(y_logits)  # binary

    # Reshape predicts and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()
