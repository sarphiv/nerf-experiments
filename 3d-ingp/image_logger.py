from typing import Any, cast
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch as th
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb

import matplotlib.pyplot as plt

from data_module import ImagePoseDataset
import os


class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        epoch_period: int,
        validation_image_name: str,
        batch_size: int,
        num_workers: int,
        path: str,
        plot_period=None
    ) -> None:
        super().__init__()
        self.logger = wandb_logger
        self.epoch_period = epoch_period
        self.validation_image_name = validation_image_name
        
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.path = path

        self.plot_period = plot_period

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if (not self.plot_period is None) and (batch_idx % self.plot_period == 0):
            self.log_image(trainer, pl_module, batch_idx)

    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # If not at the right epoch, skip
        if trainer.current_epoch % self.epoch_period == 0:
            self.log_image(trainer, model)

    @th.no_grad()
    def log_image(self, trainer: pl.Trainer, model: pl.LightningModule, batch_idx=None) -> None:
        # Retrieve validation dataset from trainer
        dataset = cast(ImagePoseDataset, trainer.datamodule.dataset_val) # type: ignore

        # Set up data loader for validation image
        data_loader = DataLoader(
            dataset=TensorDataset(
                dataset.origins[self.validation_image_name].view(-1, 3), 
                dataset.directions[self.validation_image_name].view(-1, 3)
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )


        # Iterate over batches of rays to get RGB values
        rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
        i = 0
        
        for ray_origs, ray_dirs in data_loader:
            # Prepare for model prediction
            ray_origs = ray_origs.to(model.device)
            ray_dirs = ray_dirs.to(model.device)
            
            # Get size of batch
            batch_size = ray_origs.shape[0]
            
            # Predict RGB values
            rgb[i:i+batch_size, :] = model(ray_origs, ray_dirs)[0].cpu()
        
            # Update write head
            i += batch_size


        # Log image
        # NOTE: Cannot pass tensor as channel dimension is in numpy format
        image = rgb.view(dataset.image_height, dataset.image_width, 3).detach().numpy().clip(0, 1)
        name = f"val_img_epoch={trainer.current_epoch}"
        if not batch_idx is None:
            name += f"_batch={batch_idx}"
        plt.imsave(os.path.join(self.path, f"{name}.png"), image)
        self.logger.log_image(
            key="val_img", 
            images=[image]
        )
