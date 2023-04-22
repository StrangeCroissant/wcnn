import sys
import torch
from timeit import default_timer as timer
from torchvision import datasets
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from train_test_loop import test_step, train_step
from helper_functions import accuracy_fn, cross_entropy, cross_entropy_prime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


class TileSplit(object):
    def __init__(self, tile_size=4, stride=2):
        self.tile_size = tile_size
        self.stride = stride

    def __call__(self, tensor):
        to_pil_image = transforms.ToPILImage()
        tiles = []
        for img in tensor:
            pil_img = to_pil_image(img)
            w, h = pil_img.size
            for i in range(0, w - self.tile_size + 1, self.stride):
                for j in range(0, h - self.tile_size + 1, self.stride):
                    tile = transforms.functional.crop(
                        pil_img, i, j, self.tile_size, self.tile_size
                    )
                    tiles.append(transforms.ToTensor()(tile))
        return torch.stack(tiles)

    def __shape(self, x):
        return x.shape


gc.collect()

batch_size = 32


def data_prep():
    data_trans = transforms.Compose([transforms.ToTensor(), TileSplit()])

    train_data_tiled = datasets.MNIST(
        root="./data", train=True, download=True, transform=data_trans
    )
    test_data_tiled = datasets.MNIST(
        root="./data", train=False, download=True, transform=data_trans
    )

    train_loader_tiled = torch.utils.data.DataLoader(
        train_data_tiled, batch_size=batch_size, shuffle=True
    )

    test_loader_tiled = torch.utils.data.DataLoader(
        test_data_tiled, batch_size=batch_size, shuffle=False
    )

    return train_data_tiled, test_data_tiled, train_loader_tiled, test_loader_tiled


train_data_tiled, test_data_tiled, train_loader_tiled, test_loader_tiled = data_prep()


def check_data_sizes():
    for i, (image, label) in enumerate(train_loader_tiled):
        print(f"train input image size {image.shape}\n")

        print("train label size {label.shape}\n")
        break

    for i, (image, label) in enumerate(test_loader_tiled):
        print("test input image size {image.shape}\n")
        print(label)
        print("test label size {label.shape}\n")
        break


check_data_sizes()

"""Creating the winograd class"""


class Winograd(nn.Module):
    def __init__(self, tile_size=4, stride=2):
        super(Winograd, self).__init__()
        self.tile_size = tile_size
        self.stride = stride
        self.B = torch.tensor(
            [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]],
            dtype=torch.float32,
            device=device,
        )
        self.BT = self.B.transpose(1, 0)
        self.G = torch.tensor(
            [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        self.GT = self.G.transpose(1, 0)

        self.filter = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=device
        )

        self.A = torch.tensor(
            [[1, 1, 1, 0], [0, 1, -1, -1]], dtype=torch.float32, device=device
        )
        self.AT = self.A.transpose(1, 0)

    def forward(self, x):
        transformed_tiles = []
        kernel = torch.randn(3, 3, 3)
        tiles = x.to(device)

        for tile in tiles:
            # print(tile.shape)
            D = torch.matmul(torch.matmul(self.B, tile), self.BT)
            F = torch.matmul(torch.matmul(self.G, self.filter), self.GT)
            Y = F * D
            transformed_tiles.append(Y)
        transformed_tiles = torch.stack(transformed_tiles)
        O = torch.matmul(torch.matmul(self.A, transformed_tiles), self.AT)
        gc.collect()
        return O


"""Test cnn with winograd convolution. This is the baseline to be inmprooved"""


class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.winograd = Winograd()
        self.fl1 = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(676, 10)

    def forward(self, x):
        # print('before winograd',x.shape) #before winograd torch.Size([32, 169, 1, 4, 4])
        x = self.winograd(x)
        # print('after winograd',x.shape) #after winograd torch.Size([32, 169, 1, 2, 2])
        # x = x.view(x.size(0), -1)

        x = self.fl1(x)
        # print('after flaten',x.shape) #after flaten torch.Size([32, 676])
        x = self.fc1(x)
        # print('output shape',x.shape)

        return x


gc.collect()

model = testNet()
model.to(device)

"""First training"""


def train_eval_first():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    next(model.parameters()).is_cuda

    num_epochs = 3
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize the running loss and number of correct predictions
        running_loss = 0.0
        num_correct = 0

        # Iterate over the training data in batches
        for i, (inputs, targets) in enumerate(train_loader_tiled):
            # Move the inputs and targets to the device
            inputs = inputs.to(device)

            targets = torch.tensor(targets, dtype=torch.int64).to(device)

            # print(inputs.shape)
            # Zero the gradients and perform a forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute the loss and perform a backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Update the running loss and number of correct predictions
            running_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            num_correct += torch.sum(predictions == targets.data)

            gc.collect()

            # Print the loss and accuracy every 100 stepss
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.2f"
                    % (
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        len(train_data_tiled) // 100,
                        running_loss / ((i + 1) * inputs.size(0)),
                        float(num_correct) / ((i + 1) * inputs.size(0)),
                    )
                )

        # Evaluate the model on the test data
        model.eval()
        test_loss = 0.0
        test_correct = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader_tiled):
                inputs = inputs.to(device)

                targets = torch.tensor(targets, dtype=torch.int64).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs, 1)
                test_correct += torch.sum(predictions == targets.data)
            print(
                "Test Loss: %.4f, Test Acc: %.2f"
                % (
                    test_loss / len(test_data_tiled),
                    float(test_correct) / len(test_data_tiled),
                )
            )

        gc.collect()

    return test_loss, loss, test_correct, num_correct


# train_eval_first()

"""## second training- my functions with tqdm"""


def tqdm_train_eval():
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    optimizer.zero_grad()

    epochs = 5
    train_time_start_on_device = timer()
    writer = SummaryWriter(
        "winograd/second_winograd{}".format(train_time_start_on_device)
    )

    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n-------- ")

        """Training Step """
        train_loss, train_acc = train_step(
            model=model,
            data_loader=train_loader_tiled,
            loss_fn=loss_function,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        """Testing Step """
        test_loss, test_acc = test_step(
            model=model,
            data_loader=test_loader_tiled,
            loss_fn=loss_function,
            accuracy_fn=accuracy_fn,
            device=device,
        )
        print(f"epoch {epoch} done!")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)
        gc.collect()
    train_time_end_on_device = timer()

    runtime = train_time_end_on_device - train_time_start_on_device
    print(f"Total training runntime:{runtime} second on {device}")

    return train_loss, test_loss, train_acc, test_acc


tqdm_train_eval()
# import tensorboard
