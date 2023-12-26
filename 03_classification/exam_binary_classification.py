from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import common_features.plot as cplt


def create_training_and_test_set():
    x, y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.8)
    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(x_test, dtype=torch.float32), \
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 1)
        self.activation = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x


def accuracy_fn(expected, prediction):
    # As the training going, we would like to know how well our model performs. So we create this function to calculate
    # the accuracy of the model. The accuracy is the number of correct predictions divided by the number of samples.
    correct = torch.eq(expected, prediction).sum().item()
    return correct / len(expected) * 100


def train_nn(model, x_train, y_train, learning_rate=0.2, max_epochs=10000, accuracy_threshold=99.1):
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    errors = []

    for epoch in range(max_epochs):
        model.train()
        y_pred_logits = model(x_train)
        loss = loss_fn(y_pred_logits, y_train)
        accuracy = accuracy_fn(y_train, torch.round(y_pred_logits))
        errors.append(100 - accuracy)
        if accuracy >= accuracy_threshold:
            print(f'Epoch: {epoch}, loss: {loss.item()}, accuracy: {accuracy:.2f}%')
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    cplt.plot_loss_values(torch.tensor(errors))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, x_test, y_train, y_test = create_training_and_test_set()
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    model = BinaryClassifier().to(device)
    train_nn(model, x_train, y_train)

    model.eval()
    with torch.inference_mode():
        y_test_pred = model(x_test)
        test_accuracy = accuracy_fn(y_test, torch.round(y_test_pred))
        print(f'Test accuracy: {test_accuracy:.2f}%')
        cplt.plot_decision_boundary(model, x_test, y_test)