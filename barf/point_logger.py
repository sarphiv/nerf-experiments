from typing import cast
from math import tanh, log, sqrt

import torch as th
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
import wandb
from tqdm import tqdm

from data_module import ImagePoseDataModule
from model_camera_calibration import CameraCalibrationModel



class LogCameraExtrinsics(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        logging_start: float | int,
        delay_start: float | int,
        delay_end: float | int,
        delay_taper: float | int,
        batch_size: int,
        num_workers: int,
        ray_direction_length: float,
        metric_name="train_point",
    ) -> None:
        """Log the camera extrinsics of the training set.
        
        Args:
            wandb_logger (WandbLogger): Weights and biases logger.
            validation_image_names (list[str]): Names of the validation images.
            skip_start (float | int): Epoch fraction at which logging starts.
            delay_start (float | int): Initial delay between reconstructions.
            delay_end (float | int): Final delay between reconstructions to approach.
            delay_taper (float | int): At half `delay_taper` steps, 
                the logging delay should be halfway between `delay_start` and `delay_end`.
            batch_size (int): Batch size for the data loader.
            num_workers (int): Number of workers for the data loader.
            ray_direction_length (float): Scale of the direction vectors.
            metric_name (str): Name of the metric to log.

        """
        super().__init__()

        # Verify arguments
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
        if num_workers <= 0:
            raise ValueError(f"num_workers must be positive, but is {num_workers}")
        if len(metric_name) == 0:
            raise ValueError(f"metric_name must not be empty")


        # Assign parameters
        self.logger = wandb_logger

        self.logging_start = logging_start
        self.delay_start = delay_start
        self.delay_end = delay_end
        self.delay_taper = delay_taper
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ray_direction_length = ray_direction_length
        self.metric_name = metric_name

        # Calculate next reconstruction step
        self.logging_milestone = self._get_next_delay(0)


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
        if step < self.logging_milestone:
            return

        # Update logging milestop and log
        self.logging_milestone = step + self._get_next_delay(step)


        # Retrieve validation dataset from trainer
        data_module = cast(ImagePoseDataModule, trainer.datamodule) # type: ignore
        
        
        # Transform raw camera center rays into model space
        (ray_origs_raw, ray_dirs_raw), (ray_origs_noisy, ray_dirs_noisy) = data_module.train_camera_center_rays(model.device)
        ray_origs_raw, ray_dirs_raw, _ = model.validation_transform_rays(ray_origs_raw, ray_dirs_raw)

        # Transform noisy camera center rays into model space
        ray_origs_noisy, ray_dirs_noisy, _, _ = model.camera_extrinsics.forward(
            th.arange(len(ray_origs_noisy), device=model.device), 
            ray_origs_noisy, 
            ray_dirs_noisy
        )

        # Scale directions to be visible
        ray_dirs_raw *= self.ray_direction_length
        ray_dirs_noisy *= self.ray_direction_length

        # Define colors
        # NOTE: Yellow is the color of raw rays
        ray_raw_colors = th.hstack((th.ones(len(ray_origs_raw), 2, device=model.device) * 255, th.zeros(len(ray_origs_raw), 1, device=model.device)))
        
        # NOTE: Red is wrong everything, green is correct origin, blue is correct direction
        ray_noisy_errors = 255 * th.ones(len(ray_origs_noisy), 1, device=model.device) * th.norm(ray_origs_raw - ray_origs_noisy, dim=1).view(-1, 1) / th.std(ray_origs_raw, dim=0).norm()
        ray_noisy_errors = ray_noisy_errors.clip(0, 255)
        ray_noisy_errors = th.hstack((ray_noisy_errors, 255 - ray_noisy_errors, th.zeros(len(ray_origs_noisy), 1, device=model.device)))

        # TODO: Color by ray direction error too


        # Log images
        self.logger.experiment.log({
            self.metric_name: wandb.Object3D.from_numpy(
                th.vstack((
                    th.hstack((ray_origs_raw, ray_raw_colors)),
                    th.hstack((ray_origs_noisy, ray_noisy_errors)),
                )).cpu().numpy()
            )
        })

        # self.logger.experiment.log({
        #     self.metric_name: wandb.Object3D.from_point_cloud(
        #         points=[],
        #         boxes=[],
        #         vectors=[
        #             { "start": start, "end": end}
        #             for start, end in zip(
        #                 th.vstack((
        #                     th.hstack((ray_origs_raw, ray_raw_colors)),
        #                     th.hstack((ray_origs_noisy, ray_noisy_errors)),
        #                 )).cpu().tolist(), 
        #                 th.vstack((
        #                     th.hstack((ray_origs_raw + ray_dirs_raw, ray_raw_colors)),
        #                     th.hstack((ray_origs_noisy + ray_dirs_noisy, ray_noisy_errors)),
        #                 )).cpu().tolist()
        #             )
        #         ]
        #     )
        # })
