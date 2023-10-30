import math

import torch
from torch import nn
from matplotlib import pyplot as plt
from linear_regression_model import LinearRegressionModel


class LinearRegressionRecognition:
    def __init__(self, actual_weight: float, actual_bias: float, train_set_factor: float = 0.8):
        self.y_test_set = None
        self.x_test_set = None
        self.y_train_set = None
        self.x_train_set = None

        self.actual_weight = actual_weight
        self.actual_bias = actual_bias
        self.train_set_factor = train_set_factor

    def create_data_set(self):
        x_data_set = torch.arange(0., 1., 0.01).unsqueeze(dim=1)
        y_data_set = self.actual_weight * x_data_set + self.actual_bias
        train_set_length = int(len(x_data_set) * self.train_set_factor)
        self.x_train_set = x_data_set[:train_set_length]
        self.y_train_set = y_data_set[:train_set_length]
        self.x_test_set = x_data_set[train_set_length:]
        self.y_test_set = y_data_set[train_set_length:]

    def plot_data_set(self, predictions=None):
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x_train_set, self.y_train_set, c='b', s=4, label='Train data')
        plt.scatter(self.x_test_set, self.y_test_set, c='g', s=4, label='Test data')
        if predictions is not None:
            plt.scatter(self.x_test_set, predictions, c='r', s=4, label='Predictions')

        plt.legend(prop={'size': 14})
        plt.show()

    @staticmethod
    def plot_loss_values(loss_values: torch.Tensor):
        plt.figure(figsize=(10, 7))
        plt.plot(torch.arange(0, len(loss_values), step=1), loss_values, label='Training loss')
        plt.legend(prop={'size': 14})
        plt.ylim(bottom=0)
        plt.show()

    def train_and_predict(self):
        self.create_data_set()

        # We can set the seed so that we will have the same random numbers every time we run the code. This is useful
        # for reproducibility.
        torch.manual_seed(42)
        model_0 = LinearRegressionModel()

        # If we would like to evaluate the model (check the parameter of the model), we can use `parameters()` method
        # to get all the parameters of the model. However, if we would like to get the name of the parameters, we can
        # use `state_dict()` method to get the name of the parameters, as well as the value of them.
        print('Before training, we initialized the model using random values: ', model_0.state_dict())

        # Now that we have the model (and it should have been trained already), we can use it to do the prediction.
        # If the model is not trained, of course we may get a very bad result. And this is the case for now.
        #
        # To start prediction, we need to turn the model into inference mode. And that will not track the gradient
        # and back propagation. You can do something similar using `torch.no_grad()` context manager. However, the
        # `torch.inference_mode()` is more convenient and preferred. For more information, please check the following
        # https://pytorch.org/docs/stable/generated/torch.inference_mode.html#torch.inference_mode
        with torch.inference_mode():
            predictions = model_0(self.x_test_set)
        self.plot_data_set(predictions)

        # Next we will start with the random weights and bias to do the prediction. We will look at the training data
        # and adjust the weights and bias to make the prediction more accurate.
        #
        # To do the things above. We should use 2 main algorithms:
        # 1. Gradient descent: to adjust the weights and bias to make sure the loss function is minimized. For more
        #    information, please check: https://youtu.be/IHZwWFHWa-w
        # 2. Back propagation: to calculate the gradient of the loss function with respect to the weights and bias.
        #    for more information, please check: https://youtu.be/Ilg3gGewQ5U
        #
        # While training, we need a way to measure our prediction. And one of the way is to use a loss function.
        # The loss function measures the difference between the prediction (y_pred) and the actual value (y).
        #
        # Another thing we need is an optimizer which will adjust the weights and bias during training. The optimizer
        # will use the gradient of the loss function to adjust the weights and bias. The gradient is calculated using
        # back propagation.
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model_0.parameters(), lr=0.05, weight_decay=0.0001)

        # Now we will build the training loop. The training loop will do the following things:
        # 0. Loop through the training data.
        # 1. Doing a forward pass to get the prediction (also called forward propagation).
        # 2. Calculate the loss function (compare forward predictions with the truth labels).
        # 3. Optimizer zero grad
        # 4. Perform back propagation on the loss with respect to the parameters of the model.
        # 5. Step the optimizer (perform gradient decent).
        loss_values = self.train_model(model_0, loss_function, optimizer, min_error=1e-6, max_epochs=10000)
        self.plot_loss_values(loss_values)

        # Let's check the result
        model_0.eval()
        with torch.inference_mode():
            predictions = model_0(self.x_test_set)
        self.plot_data_set(predictions)
        print(f"Model's state_dict: {model_0.state_dict()}")

    def train_model(self, model_0, loss_function, optimizer, min_error: float = 0.01, max_epochs: int = 1000):
        loss_values = []
        last_epoch = 0

        # 0. Loop through the training data.
        for epoch in range(max_epochs):
            # Set the model to training mode. Train mode in Pytorch sets all parameters requires gradients to 'require
            # gradients'. It is recommended that you always use `model.train()` when training and `model.eval()` when
            # evaluating your model. For more information please refer to:
            # https://pytorch.org/docs/stable/notes/autograd.html
            model_0.train()

            # 1. Doing a forward pass to get the prediction (also called forward propagation).
            y_prediction = model_0(self.x_train_set)

            # 2. Calculate the loss function (compare forward predictions with the truth labels).
            loss = loss_function(y_prediction, self.y_train_set)
            loss_values.append(loss)
            if loss < min_error or math.isnan(loss):
                break

            # 3. Optimizer zero grad because PyTorch will by default accumulate the gradients on subsequent backward.
            #    However, we do not require this accumulation, we only adjust parameters according to the current
            #    gradients. So we need to zero them out at each iteration for the basic gradient decent to work.
            optimizer.zero_grad()

            # 4. Perform back propagation on the loss with respect to the parameters of the model. This is to compute
            #    the gradient of every parameter with `require_grad=True`. After this, the gradient for this batch of
            #    data is accumulated into the `.grad` attribute of the parameter. And that is why we need to zero them
            #    out in step 3.
            loss.backward()

            # 5. Step the optimizer (perform gradient decent).
            # The gradient decent is looking for the minimum point for the loss function. The gradient is the slope of
            # the loss function. At first the gradient may be large, but during the training, the gradient will be
            # smaller and smaller. And that is why we need to multiply the learning rate to the gradient to make sure
            # we will not miss the minimum point. The way to dynamically adjust the learning rate is called learning
            # rate scheduling. For more information, please check: https://youtu.be/IHZwWFHWa-w.
            optimizer.step()

        return torch.tensor(loss_values, dtype=torch.float32)
