
from typing import Literal, Callable, Optional
from itertools import chain
import warnings
import math

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import nerfacc

from model_interpolation_architecture import NerfModel, PositionalEncoding
from model_interpolation import NerfInterpolationOurs, uniform_sampling_strategies, integration_strategies
from model_camera_calibration import CameraCalibrationModel

# TODO Fix such that it always uses the same model as both proposal and radiance
class MipNeRF(NerfInterpolationOurs):

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
        super().__init__(
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


class MipBarf(MipNeRF, CameraCalibrationModel):
    
    def __init__(self, start_gaussian_sigma: float, *args, **kwargs):
        CameraCalibrationModel.__init__(self, *args, **kwargs)
        self.start_gaussian_sigma = start_gaussian_sigma
        self.current_gaussian_sigma = start_gaussian_sigma
    

    def _step_helper(self, batch, batch_idx, purpose: Literal['train', 'val']):
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

        new_pixel_width = pixel_width*(max(self.current_gaussian_sigma, 1))

        # Pack batch
        batch = (
            ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            ray_colors_raw,
            img_idx,
            new_pixel_width
        )
        return super()._step_helper(batch, batch_idx, purpose)



if __name__ == "__main__":
    class dummy:
        def __init__(self) -> None:
            pass

        def hej(self,k):
            self.k = k
    
    class ko:
        def __init__(self, k):
            dummy.hej(self, k)
    
    print(ko(23).k)