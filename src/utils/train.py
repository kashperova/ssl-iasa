import os
from copy import deepcopy
from typing import Callable, Tuple, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def train_step(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable,
    device: torch.device,
) -> Tuple[float, nn.Module]:
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss = running_loss / len(train_loader)
    return loss, model

def eval_step(
    model: nn.Module,
    eval_loader: DataLoader,
    loss_fn: Callable,
    device: torch.device
) -> float:
    model.eval()
    running_loss = 0.0
    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    loss = running_loss / len(eval_loader)
    return loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable,
    device: torch.device,
    epochs: int,
    verbose: Optional[bool] = False
) -> Tuple[nn.Module, List[float], List[float]]:
    best_model, best_loss = deepcopy(model), float("inf")
    train_losses, eval_losses = [], []

    for i in tqdm(range(epochs), desc="Training"):
        train_loss, model = train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )
        train_losses.append(train_loss)

        eval_loss = eval_step(
            model=model,
            eval_loader=eval_loader,
            loss_fn=loss_fn,
            device=device
        )
        eval_losses.append(eval_loss)

        if verbose:
            print(f"Epoch [{i + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {eval_loss:.4f}")

        if eval_loss < best_loss:
            best_model = model
            best_loss = eval_loss

    return best_model, train_losses, eval_losses


def save_model(model: nn.Module, save_dir: str, save_name: str) -> None:
    torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}.pth'))


def train_test_split(
    dataset: Dataset, split_size: float
) -> Tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    train_size = int(split_size * dataset_size)
    valid_size = dataset_size - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    return train_dataset, valid_dataset
