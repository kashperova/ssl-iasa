from typing import Dict, Type
from torchvision import datasets


class Config:
    datasets: Dict[str, Type] = {
        "OxfordIIITPet": datasets.OxfordIIITPet
    }
