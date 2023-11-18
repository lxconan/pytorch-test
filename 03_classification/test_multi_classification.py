import unittest
import torch
import torch.nn as nn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# noinspection SpellCheckingInspection
import common_features.plot as cplt


RANDOM_SEEDS = 42


class MultipleClassificationTest(unittest.TestCase):
    @staticmethod
    def prepare_data(number_of_classes: int):
        x_blob, y_blob = make_blobs(
            n_samples=10000,
            n_features=2,
            centers=number_of_classes,
            cluster_std=1.5,  # standard deviation of the clusters, give the dataset some noise
            random_state=RANDOM_SEEDS   # random seed in order to reproduce the results
        )

        plt.figure(figsize=(10, 8))
        plt.title('Multi-class classification problem')
        plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.get_cmap(name='RdYlBu'), s=2)
        plt.show()

        x_blob_tensor = torch.from_numpy(x_blob).float()
        y_blob_tensor = torch.from_numpy(y_blob).long()

        x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(
            x_blob_tensor, y_blob_tensor, test_size=0.2, random_state=RANDOM_SEEDS)

        return x_blob_train, x_blob_test, y_blob_train.unsqueeze(1), y_blob_test.unsqueeze(1)

    @staticmethod
    def accuracy_fn(expected, prediction):
        # As the training going, we would like to know how well our model performs. So we create this function to
        # calculate the accuracy of the model. The accuracy is the number of correct predictions divided by the
        # number of samples.
        correct = torch.eq(expected, prediction).sum().item()
        return correct / len(expected) * 100

    def test_do_multiple_classification(self):
        x_train, x_test, y_train, y_test = self.prepare_data(4)
        model = MultiClassifier(input_features=2, output_features=4, hidden_units=8)
        torch.manual_seed(RANDOM_SEEDS)
        self.train(model, x_train, y_train)
        self.evaluate(model, x_test, y_test)

    def train(self, model, x_train, y_train, learning_rate=0.1, max_epochs=1000, accuracy_threshold=99.0):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            y_train_pred = model(x_train)
            loss = loss_function(y_train_pred, y_train.squeeze())

            accuracy = self.accuracy_fn(y_train_pred.softmax(dim=1).argmax(dim=1), y_train.squeeze())
            if accuracy > accuracy_threshold:
                print(f'Epoch: {epoch}, loss: {loss.item()}, accuracy: {accuracy:.2f}%')
                break
            loss.backward()
            optimizer.step()

    def evaluate(self, model, x_test, y_test):
        model.eval()
        with torch.inference_mode():
            y_test_pred = model(x_test)
            test_accuracy = self.accuracy_fn(y_test_pred.softmax(dim=1).argmax(dim=1), y_test.squeeze())
            print(f'Test accuracy: {test_accuracy:.2f}%')
            cplt.plot_decision_boundary(model, x_test, y_test)


class MultiClassifier(nn.Module):
    def __init__(self, input_features: int, output_features: int, hidden_units: int):
        super(MultiClassifier, self).__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


if __name__ == '__main__':
    unittest.main()
