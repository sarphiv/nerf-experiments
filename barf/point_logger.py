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
        if step < self.logging_milestone and step != 0:
            return

        # Update logging milestop and log
        self.logging_milestone = step + self._get_next_delay(step)


        # Retrieve validation dataset from trainer
        data_module = cast(ImagePoseDataModule, trainer.datamodule) # type: ignore
        
        
        # Get camera raw and noisy center origs and directions
        camera_origs_raw = data_module.dataset_train.camera_origins.to(model.device)
        camera_dirs_raw = data_module.dataset_train.camera_directions.to(model.device)
        camera_origs_noisy = data_module.dataset_train.camera_origins_noisy.to(model.device)
        camera_dirs_noisy = data_module.dataset_train.camera_directions_noisy.to(model.device)

        # Transform the raw centers to nerf coordinates 
        camera_origs_raw, camera_dirs_raw, _ = model.validation_transform_rays(camera_origs_raw, camera_dirs_raw)
        
        # Transform noisy centers origins and directions to nerf coordinates 
        camera_origs_pred, camera_dirs_pred, _, _ = model.camera_extrinsics.forward(
            list(data_module.dataset_train.index_to_index.values()), 
            camera_origs_noisy, 
            camera_dirs_noisy
        )

        # Scale directions to be visible
        camera_dirs_raw *= self.ray_direction_length
        camera_dirs_pred *= self.ray_direction_length

        n_images = data_module.dataset_train.n_images

        # Define colors
        # NOTE: Yellow is the color of raw centers
        green = th.tensor([0, 255, 0], device=model.device)
        blue = th.tensor([0, 0, 255], device=model.device)
        red = th.tensor([255, 0, 0], device=model.device)
        origins_raw_colors = blue.repeat(n_images, 1) # th.hstack((th.ones(n_images, 2, device=model.device) * 255, th.zeros(n_images, 1, device=model.device)))
        
        # NOTE: Red is wrong everything, green is correct origin, blue is correct direction
        # relic : # NOTE: See that everything that is outside one standard deviation of the original origins are red, and within a standard devition we graduate from green to red
        # origins_pred_errors = th.norm(camera_origs_raw - camera_origs_pred, dim=1).view(-1, 1) / th.std(camera_origs_raw, dim=0).norm()
        # NOTE: A point is red if it is one 10'th of the maximum distance between any two points away from the original point cloud
        origins_pred_errors = th.norm(camera_origs_raw - camera_origs_pred, dim=1).view(-1, 1)* 10 / th.cdist(camera_origs_raw, camera_origs_raw).max()
        origins_pred_errors = origins_pred_errors.clip(0, 1)
        origins_pred_colors = red * origins_pred_errors + green*(1- origins_pred_errors) #th.hstack((origins_pred_errors, 255 - origins_pred_errors, th.zeros(n_images, 1, device=model.device)))

        # Start and stop points for the direction arrows 
        direction_start_raw = camera_origs_raw
        direction_start_pred = camera_origs_pred
        direction_end_raw = camera_origs_raw + camera_dirs_raw
        direction_end_pred = camera_origs_pred + camera_dirs_pred

        # Colors for the direction arrows 
        angle = th.arccos(th.matmul(camera_dirs_pred.unsqueeze(-2), camera_dirs_raw.unsqueeze(-1)).squeeze(-1).squeeze(-1) / (th.norm(camera_dirs_pred, dim=1) * th.norm(camera_dirs_raw, dim=1)))/th.pi
        direction_colors_pred = angle.unsqueeze(-1) * red + (1 - angle).unsqueeze(-1) * green  
        direction_colors_raw = blue.repeat(n_images, 1)


        # self.logger.experiment.log({
        #     self.metric_name: wandb.Object3D.from_numpy(
        #         1 + camera_origs_raw.cpu().numpy().astype(float),
        #     )
        # })

        # Log images
        self.logger.experiment.log({
            self.metric_name: wandb.Object3D.from_point_cloud(
                points=[(*p, *c) for p,c in zip(camera_origs_raw.tolist(), origins_raw_colors.tolist())]  + [(*p, *c) for p,c in zip(camera_origs_pred.tolist(), origins_pred_colors.tolist())],
                vectors=[],
                boxes=[]
            ) 
        })
        # Log images
        # self.logger.experiment.log({
        #     self.metric_name: wandb.Object3D.from_point_cloud(
        #         points=[(*p, *c) for p,c in zip(camera_origs_raw.tolist(), origins_raw_colors.tolist())]  + [(*p, *c) for p,c in zip(camera_origs_pred.tolist(), origins_pred_colors.tolist())],
                # vectors=[{"start": start, "end": end} for start, end, color in zip(direction_start_pred.tolist(),
                #                                                                                     direction_end_pred.tolist(),
                #                                                                                     direction_colors_pred.tolist())]
                # + [{"start": start, "end": end} for start, end, color in zip(direction_start_raw.tolist(),
                #                                                                              direction_end_raw.tolist(),
                #                                                                              direction_colors_raw.tolist())],
                # vectors=[{"start": start, "end": end, "color": color} for start, end, color in zip(direction_start_pred.tolist(),
                #                                                                                     direction_end_pred.tolist(),
                #                                                                                     direction_colors_pred.tolist())]
                # + [{"start": start, "end": end, "color": color} for start, end, color in zip(direction_start_raw.tolist(),
                #                                                                              direction_end_raw.tolist(),
                #                                                                              direction_colors_raw.tolist())],
        #         vectors=[],
        #         boxes=[]
        #     ) 
        # })


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
