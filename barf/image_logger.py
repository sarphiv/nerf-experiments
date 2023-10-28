from typing import Any, cast, Optional
from math import tanh, log, sqrt

import torch as th
import torchvision as tv
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
from tqdm import tqdm

from data_module import ImagePoseDataset


class GaussianBlurScheduler(Callback):
    def __init__(self,
                 update_every_n_train_step: Optional[float] = None,
                 update_every_n_epoch: Optional[float] = None,
                 ):
        assert not (update_every_n_train_step == None and update_every_n_epoch == None), "Either update_every_n_train_step or update_every_n_eopch must be set"
        assert not (update_every_n_train_step != None and update_every_n_epoch != None), "update_every_n_train_step or update_every_n_eopch can't both be set"
        self.sigma_idx = 0
        self.update_every = cast(float, update_every_n_epoch) if update_every_n_train_step == None else cast(float, update_every_n_train_step)
        self.update_on_train_batch_start = (update_every_n_train_step != None)
        self.update_on_epoch_start = (update_every_n_epoch != None)

    # I imagine this will be rewritten.
    # the update_sigma function should be on the pl_module
    # and then it can just pick and choose from the dataloader (batch)
    # what part of it, it will use.
    # this callback should automatically set the sigma on the modul
    # and raise an error, if it does not already exist.

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        print("hej")


    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if self.update_on_train_batch_start:
            new_sigma_idx = float(trainer.global_step) // self.update_every
            if new_sigma_idx != self.sigma_idx:
                self.sigma_idx = new_sigma_idx
                trainer.datamodule.update_sigma(self.sigma_idx) #type: ignore
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.update_on_epoch_start:
            new_sigma_idx = float(trainer.global_step) // self.update_every
            if new_sigma_idx != self.sigma_idx:
                self.sigma_idx = new_sigma_idx
                trainer.datamodule.update_sigma(self.sigma_idx) #type: ignore
    


class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        validation_image_names: list[str],
        logging_start: float | int,
        delay_start: float | int,
        delay_end: float | int,
        delay_taper: float | int,
        reconstruction_batch_size: int,
        reconstruction_num_workers: int,
        metric_name="val_img",
    ) -> None:
        """Log a 2D image reconstruction of the validation image.
        
        Args:
            wandb_logger (WandbLogger): Weights and biases logger.
            validation_image_names (list[str]): Names of the validation images.
            skip_start (float | int): Epoch fraction at which logging starts.
            delay_start (float | int): Initial delay between reconstructions.
            delay_end (float | int): Final delay between reconstructions to approach.
            delay_taper (float | int): At half `delay_taper` steps, 
                the logging delay should be halfway between `delay_start` and `delay_end`.
            reconstruction_batch_size (int): Batch size for the reconstruction.
            num_workers (int): Number of workers for the data loader.
            metric_name (str): Name of the metric to log.

        """
        super().__init__()

        # Verify arguments
        if len(validation_image_names) == 0:
            raise ValueError(f"validation_image_names must not be empty")
        if logging_start < 0:
            raise ValueError(f"logging_start must be non-negative, but is {logging_start}")
        if delay_start < 0:
            raise ValueError(f"period_start must be non-negative, but is {delay_start}")
        if delay_end < 0:
            raise ValueError(f"period_end must be non-negative, but is {delay_end}")
        if delay_taper <= 0:
            raise ValueError(f"delay_taper must be positive, but is {delay_taper}")
        if delay_start > delay_end:
            raise ValueError(f"period_start must be smaller than period_end, but is {delay_start} and {delay_end}")
        if reconstruction_batch_size <= 0:
            raise ValueError(f"reconstruction_batch_size must be positive, but is {reconstruction_batch_size}")
        if reconstruction_num_workers <= 0:
            raise ValueError(f"reconstruction_num_workers must be positive, but is {reconstruction_num_workers}")
        if len(metric_name) == 0:
            raise ValueError(f"metric_name must not be empty")


        # Assign parameters
        self.logger = wandb_logger

        self.validation_image_names = validation_image_names

        self.logging_start = logging_start
        self.delay_start = delay_start
        self.delay_end = delay_end
        self.delay_taper = delay_taper
        
        self.batch_size = reconstruction_batch_size
        self.num_workers = reconstruction_num_workers
        self.metric_name = metric_name

        # Calculate next reconstruction step
        self.reconstruction_point = self._get_next_delay(0)


    def _get_next_delay(self, step: float | int) -> float:
        """Get the delay for logging.
        
        Defined by function that goes from start to end smoothly with a pseudo-delay.
           f(x) = tanh(step / delay_factor) * (end - start) + start

           Fulfills:
           f(0) = start
           f(taper/2) = (end - start) / 2

        Args:
            step (float | int): Current step.
        
        Returns:
            float: Period for logging.
        """
        # Find delay_factor such that f(delay_taper/2) = (end - start) / 2()
        delay_factor = - self.delay_taper / 2 / log(sqrt(3) / 3)
        delay = tanh(step / delay_factor) * (self.delay_end - self.delay_start) + self.delay_start

        return delay


    @th.no_grad()
    def on_train_batch_start(self, trainer: pl.Trainer, model: pl.LightningModule, batch: th.Tensor, batch_idx: int) -> None:
        # Get current step
        step = trainer.current_epoch + batch_idx/trainer.num_training_batches

        # If not yet logging, skip
        if step < self.logging_start:
            return

        # If not at the right step, return
        if step < self.reconstruction_point:
            return

        # Update reconstruction step and reconstruct
        self.reconstruction_point = step + self._get_next_delay(step)


        # Retrieve validation dataset from trainer
        dataset = cast(ImagePoseDataset, trainer.datamodule.dataset_val) # type: ignore
        
        # Store reconstructed images on CPU
        images = []


        # Reconstruct each image
        for name in tqdm(self.validation_image_names, desc="Reconstructing images", leave=False):
            # Set up data loader for validation image
            data_loader = DataLoader(
                dataset=TensorDataset(
                    dataset.origins[name].view(-1, 3), 
                    dataset.directions[name].view(-1, 3)
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False
            )

            # Iterate over batches of rays to get RGB values
            rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
            i = 0
            
            for ray_origs, ray_dirs in tqdm(data_loader, desc="Predicting RGB values", leave=False):
                # Prepare for model prediction
                ray_origs = ray_origs.to(model.device)
                ray_dirs = ray_dirs.to(model.device)
                
                # Get size of batch
                batch_size = ray_origs.shape[0]
                
                # Predict RGB values
                rgb[i:i+batch_size, :] = model(ray_origs, ray_dirs)[0].clip(0, 1).cpu()
            
                # Update write head
                i += batch_size


            # Store image on CPU
            # NOTE: Cannot pass tensor as channel dimension is in numpy format
            images.append(rgb.view(dataset.image_height, dataset.image_width, 3).numpy())


        # Log images
        self.logger.log_image(
            key=self.metric_name, 
            images=images
        )
