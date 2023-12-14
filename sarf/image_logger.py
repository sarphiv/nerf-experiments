from typing import cast, Literal
from math import tanh, log, sqrt

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from data_module import ImagePoseDataset
from model_camera_calibration import CameraCalibrationModel


class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        dataset_name: Literal["train", "val"],
        image_names: list[str],
        logging_start: float | int,
        delay_start: float | int,
        delay_end: float | int,
        delay_taper: float | int,
        batch_size: int,
        num_workers: int,
        metric_name="val_img",
    ) -> None:
        """Log a 2D image reconstruction of some image.
        
        Args:
            wandb_logger (WandbLogger): Weights and biases logger.
            dataset_name (Literal["train", "val"]): Name of the dataset to log.
            image_names (list[str]): Names of the images.
            skip_start (float | int): Epoch fraction at which logging starts.
            delay_start (float | int): Initial delay between reconstructions.
            delay_end (float | int): Final delay between reconstructions to approach.
            delay_taper (float | int): At half `delay_taper` steps, 
                the logging delay should be halfway between `delay_start` and `delay_end`.
            batch_size (int): Batch size for the reconstruction.
            num_workers (int): Number of workers for the data loader.
            metric_name (str): Name of the metric to log.

        """
        super().__init__()

        # Verify arguments
        if len(image_names) == 0:
            raise ValueError(f"image_names must not be empty")
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
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, but is {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be positive, but is {num_workers}")
        if len(metric_name) == 0:
            raise ValueError(f"metric_name must not be empty")


        # Assign parameters
        self.logger = wandb_logger

        self.dataset_name = dataset_name
        self.image_names = image_names

        self.logging_start = logging_start
        self.delay_start = delay_start
        self.delay_end = delay_end
        self.delay_taper = delay_taper
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metric_name = metric_name

        # Calculate next reconstruction step
        self.milestone = self._get_next_delay(0)


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
    def on_train_batch_start(self, trainer: pl.Trainer, model: CameraCalibrationModel, batch: th.Tensor, batch_idx: int) -> None:
        # Get current step
        step = trainer.current_epoch + batch_idx/trainer.num_training_batches

        # If not yet logging, skip
        if step < self.logging_start:
            return

        # If not at the right step, return
        if step < self.milestone:
            return

        # Update reconstruction step and reconstruct
        self.milestone = step + self._get_next_delay(step)


        # Retrieve validation dataset from trainer
        if self.dataset_name == "train":
            dataset = cast(ImagePoseDataset, trainer.datamodule.dataset_train)
        else:
            dataset = cast(ImagePoseDataset, trainer.datamodule.dataset_val)


        # Store reconstructed images on CPU
        images = []
        transform_params = None


        # Reconstruct each image
        for name in tqdm(self.image_names, desc=f"Reconstructing images ({self.metric_name})", leave=False):
            # Get rays for image
            idx = dataset.image_name_to_index[name]
            idx_underlying = dataset.index_to_index[idx] 

            if self.dataset_name == "train":
                origins = dataset.ray_origins_noisy[idx].view(-1, 3)
                directions = dataset.ray_directions_noisy[idx].view(-1, 3)
            else:
                origins = dataset.ray_origins[idx].view(-1, 3)
                directions = dataset.ray_directions[idx].view(-1, 3)


            # Set up data loader for validation image
            data_loader = DataLoader(
                dataset=TensorDataset(
                    origins, 
                    directions
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True
            )

            # Iterate over batches of rays to get RGB values
            rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
            i = 0
            
            for ray_origs, ray_dirs in tqdm(data_loader, desc="Predicting RGB values", leave=False):
                # Prepare for model prediction
                ray_origs = ray_origs.to(model.device)
                ray_dirs = ray_dirs.to(model.device)

                # Transform origins to model space
                if self.dataset_name == "train":
                    ray_origs, ray_dirs, _, _ = model.camera_extrinsics.forward(idx_underlying, ray_origs, ray_dirs)
                else:
                    ray_origs, ray_dirs, transform_params = model.validation_transform_rays(ray_origs, ray_dirs, transform_params)

                # Get size of batch
                batch_size = ray_origs.shape[0]
                
                # Predict RGB values
                rgb[i:i+batch_size, :] = model.forward(ray_origs, ray_dirs)[0].clip(0, 1).cpu()

                # Update write head
                i += batch_size


            # Store image on CPU
            # NOTE: Cannot pass tensor as channel dimension is in numpy format
            images.append(
                rgb.view(dataset.image_height, dataset.image_width, 3).detach().numpy()
            )


        # Log all images
        self.logger.log_image(
            key=self.metric_name, 
            images=images
        )
