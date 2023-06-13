import torch as th
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb


class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        epoch_period: int, 
        width: int, 
        height: int, 
        camera_to_world: th.Tensor
    ) -> None:
        super().__init__()
        self.logger = wandb_logger
        
        self.width = width
        self.height = height
        
        self.camera_to_world = camera_to_world.detach()
        
        self.epoch_period = epoch_period


    def on_validation_epoch_end(self, trainer: pl.Trainer, model: pl.LightningModule) -> None:
        # If not at the right epoch, skip
        if trainer.current_epoch % self.epoch_period != 0:
            return


        # Move projection matrix to device
        self.camera_to_world = self.camera_to_world.to(model.device)
        
        # Image coordinates
        x, y = th.meshgrid(
            th.arange(self.width, device=model.device, dtype=th.float16), 
            th.arange(self.height, device=model.device, dtype=th.float16),
            indexing="xy"
        )

        # Flatten into (w*h, 2) coordinates
        pixel_coords = th.hstack((
            x.flatten().unsqueeze(1),
            y.flatten().unsqueeze(1)
        ))

        # Get RGB values
        rgb = model(pixel_coords, self.camera_to_world).view(self.width, self.height, 3)

        # Log image
        self.logger.log_image(key="val_img", images=[rgb.cpu().numpy()])
        
        
