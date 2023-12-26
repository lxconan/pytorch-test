import random
import time

import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import math
import matplotlib.pyplot as plt
import common_features.plot as cplt


class BaselineVisionModel(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),   # Flatten the [channel, height, width] tensor to [channel * height * width] tensor
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


class NonLinearVisionModel(torch.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),   # Flatten the [channel, height, width] tensor to [channel * height * width] tensor
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)


class SimpleCnnVisionModule(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            # In our case, the input shape is [32, 1, 28, 28] (batch_size, channel, height, width), when we process
            # the data using the first Conv2d layer, the output shape will be [32, 10, 28, 28]
            # (batch_size, out_channel, height, width),
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            # ReLU will not change the shape of the tensor, so the output shape is still [32, 10, 28, 28]
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # MaxPool2d will half the height and width of the tensor, so the output shape is [32, 10, 14, 14]
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            # In our case, the input shape is [32, 10, 14, 14] (batch_size, channel, height, width), when we process
            # the data using the first Conv2d layer, the output shape will be [32, 10, 14, 14]
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # MaxPool2d will half the height and width of the tensor, so the output shape is [32, 10, 7, 7]
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 28 * 28 // 16, out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier_block(x)
        return x


def download_fashion_mnist_dataset():
    train_data = torchvision.datasets.FashionMNIST(
        root='data',        # the download folder
        train=True,         # we just want the training set
        download=True,      # download if not exists
        transform=torchvision.transforms.ToTensor(),  # how do we want to transform the data (images)
        target_transform=None   # how do we want to transform the labels/targets
    )
    test_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=False,        # we just want the test set
        download=True,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None
    )

    return train_data, test_data


def print_train_data_information(train_data):
    print('Information on training data:')
    print(f'  Number of samples: {len(train_data)}')
    print(f'  Shape of one sample image (channel, height, width): {train_data[0][0].shape}')
    print(f'  Shape of labels: {train_data.targets.shape}')
    print(f'  Labels: {train_data.class_to_idx}')


def plot_images(images_with_label, labels, number_of_images):
    # get the nearest square number of number_of_images
    square_number = math.ceil(number_of_images ** 0.5)
    images = random.choices(images_with_label, k=number_of_images)
    fig = plt.figure(figsize=(10, 10))
    for i in range(number_of_images):
        ax = fig.add_subplot(square_number, square_number, i + 1)
        ax.imshow(images[i][0].squeeze(), cmap='gray')
        ax.set_title(labels[images[i][1]])
        ax.axis('off')
    plt.show()


def train_data_in_small_batches(train_data, batch_size, epoches, device, loss_fn):
    # We use the DataLoader class to create a generator that returns a batch of data in each iteration.
    # The DataLoader class takes a dataset and a batch size as parameters.
    # The DataLoader class also has a shuffle parameter. If it is True, the generator will shuffle the data in each
    # epoch. If it is False, the generator will not shuffle the data.
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    number_of_classes = len(train_data.classes)
    model = SimpleCnnVisionModule(input_channels=1, hidden_units=15, output_shape=number_of_classes)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    for epoch in tqdm(range(epoches)):
        train_loss = 0
        print(f'Current epoch: {epoch + 1}')
        for batch, (images, labels) in enumerate(data_loader):
            model.train()
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(images)

            # Calculate loss
            loss = loss_fn(predictions, labels)
            train_loss = train_loss + loss.item()

            # Optimizer zero grad
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimizer step
            optimizer.step()

            if (batch + 1) % 500 == 0:
                print(f'Batch: {batch + 1}, loss: {loss.item()}')

    return model


def testing_data_in_small_batches(model, test_data, batch_size, device, loss_fn):
    model.to(device)
    model.eval()
    with torch.inference_mode():
        data_loader = DataLoader(test_data, batch_size=batch_size)
        success_count = 0
        total_loss = 0
        for batch in data_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            total_loss += loss_fn(predictions, labels).item()
            success_count = success_count + predictions.argmax(dim=1).eq(labels).sum().item()

        eval_result = {
            'loss': total_loss / len(test_data),
            'accuracy': success_count / len(test_data) * 100,
            'model': model.__class__.__name__,
            'device': device
        }
        print(f'Evaluation result: {eval_result}')


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = 'cuda'
    train_data, test_data = download_fashion_mnist_dataset()
    print_train_data_information(train_data)
    # plot_images(train_data, train_data.classes, 25)

    counter = time.perf_counter()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model = train_data_in_small_batches(train_data, batch_size=32, epoches=3, device=device, loss_fn=loss_fn)
    print(f'Elapsed training time on "{device}": {time.perf_counter() - counter:.2f}s')

    testing_data_in_small_batches(model, test_data, 32, device, loss_fn)
