import os
from copy import deepcopy
from typing import Callable, Union, Dict, Any, Optional, Tuple

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config.train_config import BaseTrainConfig
from datasets.base import BaseDataset
from utils.metrics import Metrics, CLASSIFICATION_TASKS
from utils.plots import plot_losses


class BaseSupervisedTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        loss_fn: Callable,
        optimizer: Optimizer,
        dataset: BaseDataset,
        config: BaseTrainConfig,
        metrics: Metrics,
        save_dir: Optional[str] = None,
        save_name: Optional[str] = "model",
    ) -> None:
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
        self.config = config
        self.metrics = metrics
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.save_name = save_name
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.train_dataset, self.eval_dataset = self.train_test_split(
            dataset, self.hyperparams["train_test_split"]
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.hyperparams["eval_batch_size"],
            shuffle=False,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparams["train_batch_size"],
            shuffle=True,
        )

        self.train_losses = []
        self.eval_losses = []

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.config.params

    def train_step(self):
        self.model.train()
        running_loss = 0.0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def eval_step(self, verbose: Optional[bool] = True):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.eval_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                if self.metrics.task in CLASSIFICATION_TASKS:
                    _, predicted = torch.max(outputs, 1)
                    self.metrics.update(labels, predicted)
                else:
                    self.metrics.update(labels, outputs)

        if verbose:
            print(str(self.metrics))

        return running_loss / len(self.eval_loader)

    def train(self, verbose: Optional[bool] = True):
        epochs, best_loss = self.hyperparams["epochs"], float("inf")

        for i in tqdm(range(epochs), desc="Training"):
            train_loss = self.train_step()
            self.train_losses.append(train_loss)

            self.metrics.reset()
            eval_loss = self.eval_step(verbose)
            self.eval_losses.append(eval_loss)

            if verbose:
                print(f"Epoch [{i + 1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {eval_loss:.4f}")

            if eval_loss < best_loss:
                best_loss = eval_loss
                self.save()

        return self.load_model()

    def eval(self):
        self.metrics.reset()
        self.eval_step()

    def load_model(self) -> nn.Module:
        self.model.load_state_dict(torch.load(os.path.join(self.save_dir, f'{self.save_name}.pth')))
        return self.model

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'{self.save_name}.pth'))

    def plot_losses(self):
        plot_losses(self.train_losses, self.eval_losses)

    @staticmethod
    def train_test_split(dataset: Dataset, split_size: float) -> Tuple[Dataset, Dataset]:
        dataset_size = len(dataset)
        train_size = int(split_size * dataset_size)
        valid_size = dataset_size - train_size

        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        return train_dataset, valid_dataset
