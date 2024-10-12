from typing import Callable, Optional, Union

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import RandAugment

from config.train_config import SLConfig
from trainers.self_learning.base import BaseSLTrainer
from datasets.augmented import AugmentedDataset
from datasets.base import BaseDataset
from utils.annotator import PseudoLabelAnnotator


class NoisyStudentTrainer(BaseSLTrainer):
    def __init__(
        self,
        teacher_model: Union[nn.Module, Callable],
        student_model: Union[nn.Module, Callable],
        loss_fn: Callable,
        labeled_dataset: BaseDataset,
        unlabeled_dataset: BaseDataset,
        teacher_optimizer: Optimizer,
        student_optimizer: Optimizer,
        teacher_lr_scheduler: LRScheduler,
        student_lr_scheduler: LRScheduler,
        config: SLConfig,
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            loss_fn=loss_fn,
            labeled_dataset=labeled_dataset,
            unlabeled_dataset=unlabeled_dataset,
            teacher_optimizer=teacher_optimizer,
            student_optimizer=student_optimizer,
            teacher_lr_scheduler=teacher_lr_scheduler,
            student_lr_scheduler=student_lr_scheduler,
            config=config,
            save_dir=save_dir,
        )
        self.augment = RandAugment(num_ops=2, magnitude=9)

    def set_student_dataset(self) -> Dataset:
        pseudo_dataset = PseudoLabelAnnotator(model=self.teacher_model).mark(
            self.unlabeled_dataset
        )
        pseudo_dataset = AugmentedDataset(dataset=pseudo_dataset, augment_fn=self.augment)
        labeled_dataset = AugmentedDataset(dataset=self.labeled_dataset, augment_fn=self.augment)

        return ConcatDataset([pseudo_dataset, labeled_dataset])

