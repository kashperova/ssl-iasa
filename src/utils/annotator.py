import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from datasets.base import BaseDataset


class PseudoLabelAnnotator:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(self.device)
        self.model.eval()

    def mark(self, dataset: Dataset) -> BaseDataset:
        inputs, pseudo_labels = [], []
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                outputs = self.model(x)
                labels = torch.argmax(outputs, dim=1)

                inputs.append(x.cpu())
                pseudo_labels.append(labels.cpu())

        inputs = torch.cat(inputs, dim=0)
        pseudo_labels = torch.cat(pseudo_labels, dim=0)

        assert inputs.size(0) == pseudo_labels.size(0), "mismatch between inputs and pseudo-labels."

        return BaseDataset(inputs=inputs, labels=pseudo_labels)
