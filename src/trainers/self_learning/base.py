import os
from copy import deepcopy
from typing import Callable, Union, Dict, Any, Optional

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import Dataset

from config.train_config import SLConfig
from datasets.base import BaseDataset
from trainers.supervised.base import BaseSupervisedTrainer


class BaseSLTrainer:
    def __init__(
            self,
            teacher_model: Union[nn.Module, Callable],
            student_model: Union[nn.Module, Callable],
            loss_fn: Callable,
            labeled_dataset: BaseDataset,
            unlabeled_dataset: BaseDataset,
            teacher_optimizer: Optimizer,
            student_optimizer: Optimizer,
            config: SLConfig,
            save_dir: Optional[str] = None,
    ) -> None:
        self.teacher_model = deepcopy(teacher_model)
        self.student_model = deepcopy(student_model)
        self.teacher_optimizer = teacher_optimizer
        self.student_optimizer = student_optimizer
        self.loss_fn = loss_fn
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.config = config
        self.save_dir = os.getcwd() if save_dir is None else save_dir

        self.teacher_trainer = BaseSupervisedTrainer(
            model=self.teacher_model,
            loss_fn=self.loss_fn,
            optimizer=self.teacher_optimizer,
            dataset=self.labeled_dataset,
            config=self.config.teacher,
            save_dir=self.save_dir
        )
        self.student_trainer = None

    @property
    def hyperparams(self) -> Dict[str, Any]:
        return self.config.params

    def set_student_dataset(self) -> Dataset:
        raise NotImplementedError

    def train_teacher(self, verbose: Optional[bool] = True):
        self.teacher_trainer.train(verbose=verbose)

    def train_student(self, verbose: Optional[bool] = True):
        if self.student_trainer is None:
            raise ValueError("Student dataset not configured")

        student_dataset = self.set_student_dataset()
        self.student_trainer = BaseSupervisedTrainer(
            model=self.teacher_model,
            loss_fn=self.loss_fn,
            optimizer=self.student_optimizer,
            dataset=student_dataset,
            config=self.config.student,
            save_dir=self.save_dir
        )
        self.student_trainer.train(verbose=verbose)

    def save_teacher(self):
        self.teacher_trainer.save("teacher")

    def save_student(self):
        self.student_trainer.save("student")

    def plot_teacher_losses(self):
        self.teacher_trainer.plot_losses()

    def plot_student_losses(self):
        self.student_trainer.plot_losses()
