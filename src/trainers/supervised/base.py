import os
from copy import deepcopy
from typing import Callable, Union, Dict, Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config.train_config import BaseTrainConfig
from datasets.base import BaseDataset
from utils.plots import plot_losses
from utils.train import train_test_split, save_model, train_model, eval_step


class BaseSupervisedTrainer:
    def __init__(
        self,
        model: Union[nn.Module, Callable],
        loss_fn: Callable,
        optimizer: Optimizer,
        dataset: BaseDataset,
        config: BaseTrainConfig,
        save_dir: Optional[str] = None,
    ) -> None:
        self.model = deepcopy(model)
        self.loss_fn = loss_fn
        self.config = config
        self.save_dir = os.getcwd() if save_dir is None else save_dir
        self.optimizer = optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset, self.eval_dataset = train_test_split(
            dataset, self.hyperparams["train_test_split"]
        )

        self.train_losses = []
        self.eval_losses = []

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.config.params

    @property
    def eval_loader(self) -> DataLoader:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.hyperparams["eval_batch_size"],
            shuffle=False,
        )

    @property
    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hyperparams["train_batch_size"],
            shuffle=True,
        )

    def train(self, verbose: Optional[bool] = True):
        best_model, self.train_losses, self.eval_losses = train_model(
            model=self.model,
            train_loader=self.train_loader,
            eval_loader=self.eval_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.hyperparams["epochs"],
            verbose=verbose,
        )
        self.model = best_model
        self.save()

    def eval(self):
        return eval_step(
            model=self.model,
            eval_loader=self.eval_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    def save(self, name: Optional[str] = "model"):
        raise save_model(self.model, self.save_dir, name)

    def plot_losses(self):
        plot_losses(self.train_losses, self.eval_losses)
