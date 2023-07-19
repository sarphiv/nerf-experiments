from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

import matplotlib.pyplot as plt

class Plotter(Callback):
    def __init__(self, path: str, plot_period=10) -> None:
        super().__init__()
        self.plot_period = plot_period
        self.path = path
        self.values = []
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int):
        self.values.append(outputs["loss"].cpu().item()) #type: ignore
        if batch_idx % self.plot_period == 0:
            plt.plot(self.values)
            plt.yscale("log")
            plt.savefig(self.path)
            plt.close()