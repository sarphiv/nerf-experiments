from typing import Literal, cast

import torch as th
from torch import nn
from model_camera_calibration import CameraCalibrationModel
from model_camera_extrinsics import CameraExtrinsics
from data_module import ImagePoseDataModule




class BarfModel(CameraCalibrationModel):
        
    @staticmethod
    def get_sigma_alpha(alpha: th.Tensor, sigma_max: float) -> th.Tensor: 
        """
        Calculate an exponential decaying sigmaa value from the alpha value.
        """
        sigma = sigma_max * 2 ** (- alpha)
        if sigma < 1/4:
            return th.tensor([0.], device=alpha.device)
        else: 
            return sigma


    ##############################################################
    # helper methods for the lightning module methods

    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        # Transform batch to model prediction space
        if purpose == "train":
            batch = self.training_transform(batch)
            
            # Update alpha (for positional encoding in BARF)
            n_batches = len(self.trainer.train_dataloader)
            epoch = self.trainer.current_epoch + batch_idx/n_batches

            self.model_radiance.position_encoder.update_alpha(epoch)
            self.model_radiance.direction_encoder.update_alpha(epoch)
            
        elif purpose == "val": batch = self.validation_transform(batch)

        # # interpolate the blurred pixel colors
        sigma = BarfModel.get_sigma_alpha(self.model_radiance.position_encoder.alpha, self.max_gaussian_sigma)

        batch = cast(ImagePoseDataModule, self.trainer.datamodule).get_blurred_pixel_colors(batch, sigma)

        # unpack batch
        (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx,
            pixel_width,
        ) = batch

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, pixel_width)


        # compute the loss
        loss_fine = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:,0])
        psnr = self.compute_psnr(loss_fine)
        pose_error = self.compute_pose_error()

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:,0])
            loss = loss_fine + loss_coarse

            # Log metrics
            log = {f"{purpose}_loss_fine": loss_fine,
                        f"{purpose}_loss_coarse": loss_coarse,
                        f"{purpose}_psnr": psnr,
                        "alpha": self.model_radiance.position_encoder.alpha.item(),
                        "sigma": sigma.item()
                        }
        else:
            loss = loss_fine
            log = {f"{purpose}_loss_fine": loss_fine,
                        f"{purpose}_psnr": psnr,
                        "alpha": self.model_radiance.position_encoder.alpha.item(),
                        "sigma": sigma.item(),
                        }

        if purpose == "train": log["pose_error"] = pose_error

        self.log_dict(log)
        
        return loss
