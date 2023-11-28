
from typing import Literal, Callable, Optional
from itertools import chain
import warnings
import math

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import nerfacc

from model_interpolation_architecture import NerfModel, PositionalEncoding
from model_interpolation import NerfInterpolation, uniform_sampling_strategies, integration_strategies
from model_camera_calibration import CameraCalibrationModel

# TODO Fix such that it always uses the same model as both proposal and radiance
class MipNeRF(NerfInterpolation):

    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        model_radiance: NerfModel,
        samples_per_ray_radiance: int,
        uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
        uniform_sampling_offset_size: float = 0.,
        integration_strategy: integration_strategies = "middle",
        samples_per_ray_proposal: int = 0,
        ):
        NerfInterpolation.__init__(self,
            self, 
            near_sphere_normalized=near_sphere_normalized,
            far_sphere_normalized=far_sphere_normalized,
            model_radiance=model_radiance,
            model_proposal=model_radiance if samples_per_ray_proposal > 0  else None,
            samples_per_ray_radiance=samples_per_ray_radiance,
            uniform_sampling_strategy=uniform_sampling_strategy,
            uniform_sampling_offset_size=uniform_sampling_offset_size,
            integration_strategy=integration_strategy,
            samples_per_ray_proposal=samples_per_ray_proposal,
        )

        self.param_groups = self.model_radiance.param_groups

    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        # unpack batch
        (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx,
            pixel_width
        ) = batch

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, pixel_width)


        # compute the loss
        loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:,0])
        psnr = -10 * math.log10(float(loss.detach().item()))
        # Log metrics
        logs  = {f"{purpose}_loss_fine": loss,
                    f"{purpose}_psnr": psnr,
                    }

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:,0])
            loss = loss + loss_coarse*0.1
            logs[f"{purpose}_loss_coarse"] = loss_coarse

        self.log_dict(logs)

        if loss.isnan():
            loss = th.tensor(1., requires_grad=True)
            print("loss was nan - no optimization step performed")

        return loss


class MipBarf(CameraCalibrationModel):
    
    def __init__(self,
            model_radiance: NerfModel,
            samples_per_ray_radiance: int,
            start_gaussian_sigma: float,
            n_training_images: int,
            camera_learning_rate_start: float,
            camera_learning_rate_stop: float,
            camera_learning_rate_decay_end: int = -1,
            max_gaussian_sigma: float = 0.0,
            near_sphere_normalized: float = 2.,
            far_sphere_normalized: float = 8.,
            uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
            uniform_sampling_offset_size: float = 0.,
            integration_strategy: integration_strategies = "middle",
            samples_per_ray_proposal: int = 0,
            gaussian_sigma_decay_start_step: int = 0,
            gaussian_sigma_decay_end_step: int = 0,
            pixel_width_follows_sigma: bool = True,
            blur_follows_sigma: bool = True,
            ):
        """
        mip barf model
            * gaussian_sigma_decay_start_step: int - When to start decaying sigma (optimization step)
            * gaussian_sigma_decay_end_step: int - when to stop decaying sigma
            * pixel_width_follows_sigma: bool - whether or not to adjust the pixel width with gaussian sigma
            * blur_follows_sigma: bool - wether or not to actually blur the image
        
        Details:
        ----------
        the gaussian sigma follows the following schedule:

            * current step < start step: sigma = start sigma
            * start step < current step < end step: sigma follows exponential decay from start sigma at start step down to 1/4 at stop step
            * end step < current step: sigma = 0.

        pixel_width_follows_sigma and blur_follows_sigma are made to give control and make ablations possible.
        For example if you wish to run an experiment where you only down scale positional encodings
        accoding to the sigma schedule (gaussian_sigma_decay_start_step and gaussian_sigma_decay_end_step)
        but not actually blur the image (similar to what barf does) - just set pixel_width_follows_sigma=True and
        blur_follows_sigma = False.




        For the following arguments, see NerfInterpolation and CameraCalibrationModel:
            * model_radiance
            * samples_per_ray_radiance
            * start_gaussian_sigma
            * n_training_images
            * camera_learning_rate_start
            * camera_learning_rate_stop
            * camera_learning_rate_decay_end
            * max_gaussian_sigma [not used]
            * near_sphere_normalized
            * far_sphere_normalized
            * uniform_sampling_strategy
            * uniform_sampling_offset_size
            * integration_strategy
            * samples_per_ray_proposal
        """
        
        CameraCalibrationModel.__init__(self,
            model_radiance=model_radiance,
            model_proposal=model_radiance if samples_per_ray_proposal > 0 else None,
            samples_per_ray_radiance=samples_per_ray_radiance,
            n_training_images=n_training_images,
            camera_learning_rate_start=camera_learning_rate_start,
            camera_learning_rate_stop=camera_learning_rate_stop,
            camera_learning_rate_decay_end=camera_learning_rate_decay_end,
            max_gaussian_sigma=max_gaussian_sigma,
            near_sphere_normalized=near_sphere_normalized,
            far_sphere_normalized=far_sphere_normalized,
            uniform_sampling_strategy=uniform_sampling_strategy,
            uniform_sampling_offset_size=uniform_sampling_offset_size,
            integration_strategy=integration_strategy,
            samples_per_ray_proposal=samples_per_ray_proposal,
            )
        
        self.start_gaussian_sigma            = start_gaussian_sigma
        self.gaussian_sigma_decay_start_step = gaussian_sigma_decay_start_step
        self.gaussian_sigma_decay_end_step   = gaussian_sigma_decay_end_step

        self.register_buffer("sigma_schedule", th.tensor(start_gaussian_sigma))
        self.pixel_width_follows_sigma = pixel_width_follows_sigma
        self.blur_follows_sigma = blur_follows_sigma
    
        self.param_groups = [param_group for model in [self.model_radiance, self.camera_extrinsics]
                             for param_group in model.param_groups]


    def update_sigma_schedule(self, current_step):
        if current_step < self.gaussian_sigma_decay_start_step:
            sigma_schedule = self.start_gaussian_sigma

        elif self.gaussian_sigma_decay_start_step <= current_step <= self.gaussian_sigma_decay_end_step:
            sigma_schedule = (
                self.start_gaussian_sigma * (
                    self.start_gaussian_sigma/0.25
                )**(
                    (
                        self.gaussian_sigma_decay_start_step - current_step
                    )/(
                        self.gaussian_sigma_decay_start_step - self.gaussian_sigma_decay_end_step
                    )
                )

            )

        else:
            sigma_schedule = 0.
        
        self.sigma_schedule = th.tensor(sigma_schedule, device=self.sigma_schedule.device)


    # TODO should be specified from main - to allow experiments with different schedules.
    @property
    def pixel_width_scalar(self):
        if self.pixel_width_follows_sigma:
            return (12*self.current_gaussian_sigma**2 + 1)**0.5
        else:
            return 1.

    def _step_helper(self, batch, batch_idx, purpose: Literal['train', 'val']):

        if purpose == "train":
            n_batches = len(self.trainer.train_dataloader)
            current_step = self.trainer.current_epoch*n_batches + batch_idx
            self.update_sigma_schedule(current_step)

            batch = self.training_transform(batch)
        elif purpose == "val":
            batch = self.validation_transform(batch)
        else:
            raise ValueError(f"purpose={purpose} is invalid")
        # Unpack batch
        (
            ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            ray_colors_raw,
            img_idx,
            pixel_width
        ) = self.trainer.datamodule.get_blurred_pixel_colors(batch, self.current_gaussian_sigma)

        new_pixel_width = pixel_width*self.pixel_width_scalar

        pose_error = CameraCalibrationModel.compute_pose_error(self)

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, new_pixel_width)


        if self.blur_follows_sigma: ray_colors_true = ray_colors_raw[:,0]
        else:                       ray_colors_true = ray_colors_raw[:,-1]

        # compute the loss
        loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_true)
        psnr = -10 * math.log10(float(loss.detach().item()))
        # Log metrics
        logs  = {f"{purpose}_loss_fine": loss,
                f"{purpose}_psnr": psnr,
                    }

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_true)
            loss = loss + loss_coarse*0.1
            logs[f"{purpose}_loss_coarse"] = loss_coarse

        if purpose == "train": logs["train_pose_errors"] = pose_error
        self.log_dict(logs)

        if loss.isnan():
            loss = th.tensor(1., requires_grad=True)
            print("loss was nan - no optimization step performed")

        return loss