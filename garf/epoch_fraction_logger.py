from typing import cast

import torch as th
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore

from data_module import ImagePoseDataset


class LogEpochFraction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        metric_name: str = "epoch_fraction",
    ) -> None:
        """Log a fractional epoch that takes batches into account
        
        Args:
            wandb_logger (WandbLogger): Weights and biases logger.
            metric_name (str): Name of the metric to log.
        """
        super().__init__()
        self.logger = wandb_logger
        self.metric_name = metric_name


    @th.no_grad()
    def on_train_batch_start(self, trainer: pl.Trainer, model: pl.LightningModule, batch: th.Tensor, batch_idx: int) -> None:
        # Log epoch fraction
        self.logger.log_metrics({
            self.metric_name: trainer.current_epoch + batch_idx/trainer.num_training_batches
        })