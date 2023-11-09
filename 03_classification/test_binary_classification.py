import unittest
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import common_features.plot as cplt
from common_features.plot import plot_decision_boundary


class CircleClassifier(nn.Module):
    def __init__(self):
        super(CircleClassifier, self).__init__()
        self.layer_input = nn.Linear(2, 10)
        self.layer_hidden = nn.Linear(10, 10)
        self.layer_output = nn.Linear(10, 1)
        self.activation = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer_input(x)
        x = self.activation(x)
        x = self.layer_hidden(x)
        x = self.activation(x)
        x = self.layer_output(x)
        return self.sigmoid(x)


def accuracy_fn(expected, prediction):
    # As the training going, we would like to know how well our model performs. So we create this function to calculate
    # the accuracy of the model. The accuracy is the number of correct predictions divided by the number of samples.
    correct = torch.eq(expected, prediction).sum().item()
    return correct / len(expected) * 100


class TestBinaryClassification(unittest.TestCase):
    def test_should_do_binary_classification(self):
        n_samples = 1000
        # The function make_circles generates a binary classification problem with 2 features. It will create a big
        # circle containing half of the samples and a smaller circle containing the other half of the samples.
        #
        # The noise parameter controls the amount of noise in the data. Make the circle not perfect.
        #
        # The positions parameter contains the point in the circles. Its shape is [n_samples, 2]. where each row
        # contains 2 elements, the first is the X coordinate and the second is the Y coordinate. While the circle_id
        # contains the class of each point. For example, 0 means the first circle and 1 means the second circle.
        positions, circle_id = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

        plt.figure(figsize=(10, 8))
        plt.title('Binary classification problem')
        plt.scatter(positions[:, 0], positions[:, 1], c=circle_id, cmap=plt.cm.RdYlBu)
        plt.show()

        # Since we will use tensor in PyTorch, we need to convert the positions and circle_id to tensor.
        positions = torch.from_numpy(positions).float()
        circle_id = torch.from_numpy(circle_id).float()

        # Now we will leverage sklearn to split the data into train and test sets. We will use 20% of the data for
        # testing and 80% for training. Sklearn will shuffle the data before splitting it rather than do it
        # sequentially. It accepts arrays or tensors as input.
        positions_train, positions_test, circle_id_train, circle_id_test = train_test_split(
            positions, circle_id,
            test_size=0.2,
            random_state=42)
        circle_id_train = circle_id_train.unsqueeze(1)
        circle_id_test = circle_id_test.unsqueeze(1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = CircleClassifier().to(device)
        positions_train = positions_train.to(device)
        positions_test = positions_test.to(device)
        circle_id_train = circle_id_train.to(device)
        circle_id_test = circle_id_test.to(device)

        self.train(model, positions_train, circle_id_train, epochs=10000, accuracy_threshold=99.9)

        # Test
        with torch.inference_mode():
            model.eval()
            circle_id_pred = model(positions_test)
            circle_id_pred = torch.round(circle_id_pred)
            accuracy = accuracy_fn(circle_id_test, circle_id_pred)
            print(f'Accuracy: {accuracy}')
            plot_decision_boundary(model, positions_test, circle_id_test)

    @staticmethod
    def train(model, positions_train, circle_id_train, epochs: int, accuracy_threshold: float):
        # We will use the binary cross entropy loss function. It is used for binary classification problems. It
        # expects the output of the model to be a single value between 0 and 1. It will compare this value with the
        # target value. The target value is either 0 or 1. The loss function will return a value that represents the
        # error between the output and the target. The lower the value, the better the model.
        #
        # The BCEWithLogitsLoss is a combination of the sigmoid function and the binary cross entropy loss function.
        # It is more numerically stable than using the sigmoid function and the binary cross entropy loss function.
        # In more complex cases, we may want to separate the sigmoid function and the binary cross entropy loss to
        # get more control over the model. But in this simple case we will use the BCEWithLogitsLoss.
        loss_function = nn.BCEWithLogitsLoss()

        # The SGD and Adam optimizer are suitable for regression problem and/or classification problems.
        optimizer = optim.SGD(params=model.parameters(), lr=0.2)

        errors = []
        for epoch in range(0, epochs):
            # Switch the model to train mode.
            model.train()

            # Do some predictions.
            circle_id_pred_logits = model(positions_train)
            circle_id_pred = torch.round(circle_id_pred_logits)  # Convert the output to 0 or 1.

            # Calculate loss and accuracy
            loss = loss_function(circle_id_pred_logits, circle_id_train)
            accuracy = accuracy_fn(circle_id_train, circle_id_pred)
            errors.append(100 - accuracy)
            if accuracy > accuracy_threshold:
                print(f'Stop training since accuracy matched: Epoch: {epoch}, accuracy: {accuracy}')
                break

            # Clear accumulated grad
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Update weights
            optimizer.step()

        cplt.plot_loss_values(torch.tensor(errors))



if __name__ == '__main__':
    unittest.main()
