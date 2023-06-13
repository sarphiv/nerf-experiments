from typing import cast

import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb

from data_module import ImageSyntheticDataset


class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        validation_image_name: str,
        epoch_period: int, 
        width: int, 
        height: int
    ) -> None:
        super().__init__()
        self.logger = wandb_logger
        self.validation_image_name = validation_image_name
        
        self.width = width
        self.height = height

        self.epoch_period = epoch_period


    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # If not at the right epoch, skip
        if trainer.current_epoch % self.epoch_period != 0:
            return

        # Retrieve validation dataset from trainer
        dataset = cast(ImageSyntheticDataset, trainer.datamodule.dataset_val) # type: ignore

        # TODO: Batchify rays

        # Get rays
        ray_origs = dataset.origins[self.validation_image_name].to(model.device).view(-1, 3)
        ray_dirs = dataset.directions[self.validation_image_name].to(model.device).view(-1, 3)

        # Get RGB values
        rgb = model(ray_origs, ray_dirs).view(self.height, self.width, 3)

        # Log image
        self.logger.log_image(key="val_img", images=[rgb.cpu().numpy()])
