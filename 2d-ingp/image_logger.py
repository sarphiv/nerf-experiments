import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb
import matplotlib.pyplot as plt
import os


class Log2dImageReconstruction(Callback):
    def __init__(self, wandb_logger: WandbLogger, epoch_period: int, width: int, height: int, path: str) -> None:
        super().__init__()
        self.logger = wandb_logger
        
        self.width = width
        self.height = height
        self.resolution = min(width, height)

        self.path = path
        
        self.epoch_period = epoch_period


    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # If not at the right epoch, skip
        if trainer.current_epoch % self.epoch_period != 0:
            return

        x, y = th.meshgrid(
            th.arange(self.resolution), 
            th.arange(self.resolution),
            indexing="ij"
        )
        x, y = x.flatten(), y.flatten()

        location = th.hstack((
            x.float().unsqueeze(1) / self.resolution,
            y.float().unsqueeze(1) / self.resolution
        ))

        location += 1/2/self.resolution


        # Get RGB values
        rgb = model(location.to(model.device)).view(self.width, self.height, 3).permute(1, 0, 2)

        image = rgb.cpu().numpy()

        plt.imsave(os.path.join(self.path, f"output_image_epoch={trainer.current_epoch}.png"), image)


        # Log image
        self.logger.log_image(key="val_img", images=[image])
        
        
