from typing import List

import numpy as np
import torchvision
import plotly.graph_objects as go

from matplotlib import pyplot as plt
from torch import Tensor


def plot_image(image: Tensor):
    image = torchvision.utils.make_grid(image[:4])
    image = image / 2 + 0.5
    image = image.numpy()
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()


def plot_losses(train_losses: List[float], eval_losses: List[float]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode="lines", name="Training Loss"))
    fig.add_trace(go.Scatter(y=eval_losses, mode="lines", name="Validation Loss"))
    fig.update_layout(
        title="Losses",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend=dict(x=0, y=1, traceorder="normal"),
    )
    fig.show()
