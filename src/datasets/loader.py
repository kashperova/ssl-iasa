import pathlib
from typing import Optional, Callable, Union

from torch.utils.data import Dataset

from config.datasets_config import Config


class DatasetLoader:
    dataset = None

    def __init__(
        self,
        name: str,
        root_dir: Union[str, pathlib.Path] = "./data",
        transform: Optional[Callable] = None,
    ):
        self._name = name
        self._root_dir = root_dir
        self._transform = transform

    def load(self) -> Dataset:
        self.dataset = Config.datasets[self._name](
            root=self._root_dir, download=True, transform=self._transform
        )
        return self.dataset
