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


class LogRay(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        dataset_name: Literal["train", "val"],
        image_names: list[str],
        samples_per_ray: int,
        logging_start: float | int,
        delay_start: float | int,
        delay_end: float | int,
        delay_taper: float | int,
        metric_name="val_ray",
    ) -> None:
        """Log rays of the validation image.
        
        Args:
            wandb_logger (WandbLogger): Weights and biases logger.
            dataset_name (Literal["train", "val"]): Name of the dataset to log.
            image_names (list[str]): Names of the images to use rays from.
            samples_per_ray (int): Number of samples per ray.
            skip_start (float | int): Epoch fraction at which logging starts.
            delay_start (float | int): Initial delay between reconstructions.
            delay_end (float | int): Final delay between reconstructions to approach.
            delay_taper (float | int): At half `delay_taper` steps, 
                the logging delay should be halfway between `delay_start` and `delay_end`.
            metric_name (str): Name of the metric to log.

        """
        super().__init__()

        # Verify arguments
        if len(image_names) == 0:
            raise ValueError(f"image_names must not be empty")
        if samples_per_ray <= 0:
            raise ValueError(f"samples_per_ray must be positive, but is {samples_per_ray}")
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
        if len(metric_name) == 0:
            raise ValueError(f"metric_name must not be empty")


        # Assign parameters
        self.logger = wandb_logger

        self.dataset_name = dataset_name
        self.image_names = image_names
        self.samples_per_ray = samples_per_ray

        self.logging_start = logging_start
        self.delay_start = delay_start
        self.delay_end = delay_end
        self.delay_taper = delay_taper

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


        # Store reconstructed ray plots on CPU
        ray_plots = []
        transform_params = None
        
        # Examine rays of images
        for name in tqdm(self.image_names, desc=f"Reconstructing rays ({self.metric_name})", leave=False):
            # Get rays for image
            idx = dataset.image_name_to_index[name]
            idx_underlying = th.tensor([dataset.index_to_index[idx]]).to(model.device).view(1, 1)
            ray_orig_center = dataset.camera_origins[idx].to(model.device).view(1, 3)
            ray_dir_center = dataset.camera_directions[idx].to(model.device).view(1, 3)

            # Transform rays
            if self.dataset_name == "train":
                ray_orig_center, ray_dir_center, _, _ = model.camera_extrinsics.forward(
                    idx_underlying, 
                    ray_orig_center, 
                    ray_dir_center
                )
            else:
                ray_orig_center, ray_dir_center, transform_params = model.validation_transform_rays(
                    ray_orig_center, 
                    ray_dir_center, 
                    transform_params
                )


            bin_width = (model.far_plane - model.near_plane)/self.samples_per_ray
            
            # Compute the intervals for the center ray
            center_ray_t_start = th.linspace(
                model.near_plane, 
                model.far_plane, 
                self.samples_per_ray, 
                device=model.device
            )

            center_ray_t_end = center_ray_t_start + bin_width
            
            # Compute the positions for the given t values
            pos_samples = model._get_positions(
                ray_orig_center, 
                ray_dir_center, 
                center_ray_t_start.view(1,-1), 
                center_ray_t_end.view(1,-1)
            )

            # Evaluate density and color at sample positions
            radiance_color, radiance_density = model.radiance_network.forward(
                pos_samples.view(-1, 3),
                ray_dir_center.broadcast_to(pos_samples.shape).view(-1, 3)
            )

            proposal_density = model.proposal_network.forward(
                pos_samples.view(-1, 3)
            )


            # Make a Figure and attach it to a canvas.
            fig = Figure(figsize=(5, 4), dpi=300)
            canvas = FigureCanvasAgg(fig)

            # Plot the center ray
            ax = fig.add_subplot(111)

            x = center_ray_t_start.view(-1).cpu().detach().numpy()
            col = radiance_color.view(-1, 3).cpu().detach().numpy()
            y = radiance_density.view(-1).cpu().detach().numpy()
            y2 = proposal_density.view(-1).cpu().detach().numpy()
            y_max = max(y.max(), y2.max())

            # Plot vertical bars with adjusted width
            for xi, radiance_color in zip(x, col):
                ax.bar(
                    xi, 
                    y_max*1.1, 
                    color=radiance_color, 
                    alpha=1., 
                    align='edge', 
                    width=bin_width*1.1
                )

            # Plot the density graph
            ax.plot(x, y, color='red', markersize=0.3, label = "Radiance")
            ax.plot(x, y2, color='green', markersize=0.3, label = "Proposal")


            # Get handles for data that have already been plotted to axis
            handles, labels = ax.get_legend_handles_labels()

            # Manually define a new patch 
            patch = mpatches.Patch(color=[100/255,75/255,0/255], label='Color')

            # Append manual color patch
            handles.append(patch) 

            # Set axis labels
            ax.set_xlabel("t")
            ax.set_ylabel("Density")
            ax.legend(handles=handles)
            ax.set_title(f"Center ray ({name})")

            # Retrieve a view on the render buffer
            canvas.draw()
            buf = canvas.buffer_rgba()


            # Convert to NumPy array and store for logging
            ray_plots.append(np.asarray(buf))


        # Log all plots
        self.logger.log_image(
            key=self.metric_name, 
            images=ray_plots
        )

