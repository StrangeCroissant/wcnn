from helper_functions import accuracy_fn
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.cuda.amp import autocast, GradScaler
# ignore noisy warnings
warnings.filterwarnings("ignore")
# device agnostic script runs in cuda if able
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn,
    device: torch.device = device,
):
    # inititalize 0 acc and loss
    train_loss, train_acc = 0, 0

    # set model to train mode
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        # move data to device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)
        # loss_per_batch
        loss = loss_fn(y_pred, y)

        # accumulate loss and accc
        train_loss += loss

        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        # zero grad optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        

    # train loss acc average per batch

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss:{train_loss:.5f} | Train acc:{train_acc:.2f}%")
    return train_loss, train_acc


def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    accuracy_fn,
    device: torch.device = device,
):
    """
    Performs testing and evaluation on trained model_x

    """
    test_loss, test_acc = 0, 0
    # Put the model on eval mode
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            # sent to device
            X, y = X.to(device), y.to(device)

            # forward
            test_pred = model(X)

            # calculate loss
            test_loss += loss_fn(test_pred, y)

            # calulate acc
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        # calc test loss average per batch
        test_loss /= len(data_loader)

        # calc acc average per batch

        test_acc /= len(data_loader)
    # print out evals
    print(f"Test loss:{test_loss:.5f} | Test acc:{test_acc:.2f}% \n")
    return test_loss, test_acc
