from typing import cast
from math import tanh, log, sqrt

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
import torch as th
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger #type: ignore
from tqdm import tqdm

from data_module import ImagePoseDataset
from model_camera_calibration import NerfInterpolation


# TODO: maybe fix image logger such that the model handles more of the work
#       Ideally we should just be able to ask the model for an image, with a camera to world matrix,
#       and it should return the image.
#       This would also make it easier to use the model for other things, like rendering a video.
#       This should probably make use of the functions in the data module, such that we can use the
#       as they are already implemented there. - (such as meshgrid e.g.)
#       Ideally it should be put in the NerfInterpolation class as it is a functionality that should
#       be model agnostic preferably.

# TODO: Enable the logging of training images as well

class Log2dImageReconstruction(Callback):
    def __init__(
        self, 
        wandb_logger: WandbLogger, 
        validation_image_names: list[str],
        train_image_names: list[str],
        logging_start: float | int,
        delay_start: float | int,
        delay_end: float | int,
        delay_taper: float | int,
        reconstruction_batch_size: int,
        reconstruction_num_workers: int,
        metric_name_val="val_img",
        metric_name_train="train_img"
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
        if reconstruction_num_workers < 0:
            raise ValueError(f"reconstruction_num_workers must be positive, but is {reconstruction_num_workers}")
        if len(metric_name_val) == 0:
            raise ValueError(f"metric_name must not be empty")


        # Assign parameters
        self.logger = wandb_logger

        self.validation_image_names = validation_image_names
        self.train_image_names = train_image_names

        self.logging_start = logging_start
        self.delay_start = delay_start
        self.delay_end = delay_end
        self.delay_taper = delay_taper
        
        self.batch_size = reconstruction_batch_size
        self.num_workers = reconstruction_num_workers
        self.metric_name_val = metric_name_val
        self.metric_name_train = metric_name_train

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
    def on_train_batch_start(self, trainer: pl.Trainer, model: NerfInterpolation, batch: th.Tensor, batch_idx: int) -> None:
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
        val_images = []
        val_images_true = []

        ray_plots = []

        transform_params = None
        
        # Reconstruct each image
        for name in tqdm(self.validation_image_names, desc="Reconstructing images", leave=False):
            # Get rays for image
            origins = dataset.ray_origins[dataset.image_name_to_index[name]].view(-1, 3)
            directions = dataset.ray_directions[dataset.image_name_to_index[name]].view(-1, 3)
            images = dataset.images[dataset.image_name_to_index[name]]
            images = images.view(-1, *images.shape[2:])

            samples_per_center_ray = 100

            ##### log and visualize the center ray of the image
            center_ray_origin = dataset.camera_origins[dataset.image_name_to_index[name]].to(model.device).view(1, 3)
            center_ray_direction = dataset.camera_directions[dataset.image_name_to_index[name]].to(model.device).view(1, 3)

            ray_origs, ray_dirs, transform_params = model.validation_transform_rays(center_ray_origin, center_ray_direction, None)


            near = model.near_sphere_normalized
            far = model.far_sphere_normalized
            samples_per_ray = model.samples_per_ray_radiance
            sample_bin_width = (far - near)/samples_per_ray

            # Compute the t values for the center ray
            center_ray_t_start = th.linspace(near, far, samples_per_center_ray, device=model.device)
            center_ray_t_end = center_ray_t_start + sample_bin_width

            # # Compute the positions for the given t values
            sample_pos, sample_dir = NerfInterpolation._compute_positions(model, center_ray_origin.view(1,3), center_ray_direction.view(1,3), center_ray_t_start.view(1,-1), center_ray_t_end.view(1,-1))
            sample_pixel_width = dataset.pixel_width*th.ones(sample_pos.shape[1], 1, device=model.device)


            # Evaluate density and color at sample positions
            sample_density, sample_color = model.model_radiance.forward(sample_pos.squeeze(), sample_dir.squeeze(), sample_pixel_width, center_ray_t_start.unsqueeze(1), center_ray_t_start.unsqueeze(1))

            # make a Figure and attach it to a canvas.
            fig = Figure(figsize=(5, 4), dpi=300)
            # fig = plt.figure()
            canvas = FigureCanvasAgg(fig)

            # Do some plotting here
            ax = fig.add_subplot(111)

            x = center_ray_t_start.view(-1).cpu().detach().numpy()
            col = sample_color.view(-1, 3).cpu().detach().numpy()
            y = sample_density.view(-1).cpu().detach().numpy()

            bin_width = (far - near)/samples_per_center_ray
            # Plot vertical bars with adjusted width
            for xi, color in zip(x, col):
                ax.bar(xi, max(y)*1.1, color=color, alpha=1., align='edge', width=bin_width*1.1)

            # Plot the density graph
            ax.plot(x, y, color='black', markersize=0.3, label = "Volumetric density")


            # where some data has already been plotted to ax
            handles, labels = ax.get_legend_handles_labels()

            # manually define a new patch 
            patch = mpatches.Patch(color=[100/255,75/255,0/255], label='Color')

            # handles is a list, so append manual patch
            handles.append(patch) 


            ax.set_xlabel("t")
            ax.set_ylabel("density")
            ax.legend(handles=handles)
            ax.set_title("Density and color predictions along the center ray")

            # Retrieve a view on the renderer buffer
            canvas.draw()
            buf = canvas.buffer_rgba()
            # convert to a NumPy array
            X = np.asarray(buf)

            # Store image on CPU
            ray_plots.append(X)       
            
            ##### end of ray plot

            #### log and visualize the image

            # Set up data loader for validation image
            data_loader = DataLoader(
                dataset=TensorDataset(
                    origins, 
                    directions,
                    dataset.pixel_width*th.ones(origins.shape[0], 1),
                    images
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True
            )

            # Iterate over batches of rays to get RGB values
            rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
            rgb_true = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
            i = 0
            
            for ray_origs, ray_dirs, pixel_width, ray_colors_true in tqdm(data_loader, desc="Predicting RGB values", leave=False):
                # Prepare for model prediction
                ray_origs = ray_origs.to(model.device)
                ray_dirs = ray_dirs.to(model.device)
                pixel_width = pixel_width.to(model.device)

                # Transform origins to model space
                ray_origs, ray_dirs, transform_params = model.validation_transform_rays(ray_origs, ray_dirs, transform_params)

                # Get size of batch
                batch_size = ray_origs.shape[0]
                
                # Predict RGB values
                rgb[i:i+batch_size, :] = model.forward(ray_origs, ray_dirs, pixel_width)[0].clip(0, 1).cpu()
            

                # load true image
                if hasattr(model, "current_blur_sigma"):
                    ray_colors_true = trainer.datamodule.get_blurred_pixel_colors(
                        (None, None, None, None, ray_colors_true, None, None), model.current_blur_sigma
                    )[4]
                

                rgb_true[i:i+batch_size, :] = ray_colors_true[:, 0]
                # Update write head
                i += batch_size


            # Store image on CPU
            # NOTE: Cannot pass tensor as channel dimension is in numpy format
            val_images.append(rgb.view(dataset.image_height, dataset.image_width, 3).detach().numpy())
            val_images_true.append(rgb_true.view(dataset.image_height, dataset.image_width, 3).detach().numpy())


        # Reconstruct training images 
        # Retrieve training dataset from trainer
        train_dataset = cast(ImagePoseDataset, trainer.datamodule.dataset_train) # type: ignore
        
        # Store reconstructed images on CPU
        train_images = []

        for name in tqdm(self.train_image_names, desc="Reconstructing training images", leave=False):
            # Get rays for image
            idx = train_dataset.image_name_to_index[name]
            origins_noisy = train_dataset.ray_origins_noisy[idx].view(-1, 3)
            directions_noisy = train_dataset.ray_directions_noisy[idx].view(-1, 3)
            index = train_dataset.index_to_index[idx] # gets the index of the image in the dataset - passed to model.camera_extrinsics.forward
            
            # Set up data loader for validation image
            data_loader = DataLoader(
                dataset=TensorDataset(
                    origins_noisy, 
                    directions_noisy,
                    dataset.pixel_width*th.ones(origins.shape[0], 1)
                ),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True
            )

            # Iterate over batches of rays to get RGB values
            rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
            transform_params = None
            i = 0
            
            for ray_origs, ray_dirs, pixel_width in tqdm(data_loader, desc="Predicting RGB values", leave=False):
                # Prepare for model prediction
                ray_origs = ray_origs.to(model.device)
                ray_dirs = ray_dirs.to(model.device)
                pixel_width = pixel_width.to(model.device)

                # Transform origins to model space
                ray_origs, ray_dirs, _, _ = model.camera_extrinsics(index, ray_origs, ray_dirs)

                # Get size of batch
                batch_size = ray_origs.shape[0]
                
                # Predict RGB values
                model_out = model.forward(ray_origs, ray_dirs, pixel_width)[0]
                rgb[i:i+batch_size, :] = model_out.clip(0, 1).cpu()
            
                # Update write head
                i += batch_size


            # Store image on CPU
            # NOTE: Cannot pass tensor as channel dimension is in numpy format
            train_images.append(rgb.view(dataset.image_height, dataset.image_width, 3).numpy())

        # Log images
        self.logger.log_image(
            key=self.metric_name_train, 
            images=train_images
        )
        
        self.logger.log_image(
            key=self.metric_name_val, 
            images=val_images
        )

        self.logger.log_image(
            key=f"{self.metric_name_val}_target", 
            images=val_images_true
        )
        self.logger.log_image(
            key=f"{self.metric_name_val}_center_ray", 
            images=ray_plots
        )
