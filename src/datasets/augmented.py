from typing import Callable

from torch.utils.data import Dataset

from datasets.base import BaseDataset


class AugmentedDataset(Dataset):
    def __init__(self, dataset: BaseDataset, augment_fn: Callable):
        self.dataset = dataset
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        augmented = self.augment_fn(data)
        return augmented, label
