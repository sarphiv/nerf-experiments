import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb


class Log2dImageReconstruction(Callback):
    def __init__(self, wandb_logger: WandbLogger, epoch_period: int, width: int, height: int) -> None:
        super().__init__()
        self.logger = wandb_logger
        
        self.width = width
        self.height = height
        
        self.epoch_period = epoch_period


    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # If not at the right epoch, skip
        if trainer.current_epoch % self.epoch_period != 0:
            return

        # Image coordinates
        x, y = th.meshgrid(
            th.linspace(0, 1, self.width, device=model.device), 
            th.linspace(0, 1, self.height, device=model.device),
            indexing="ij"
            )

        # Flatten into (w*h, 2) coordinates
        location = th.hstack((
            x.flatten().unsqueeze(1),
            y.flatten().unsqueeze(1)
        ))

        # Get RGB values
        rgb = model(location).view(self.width, self.height, 3).permute(1, 0, 2)

        # Log image
        self.logger.log_image(key="val_img", images=[rgb.cpu().numpy()])
        
        
