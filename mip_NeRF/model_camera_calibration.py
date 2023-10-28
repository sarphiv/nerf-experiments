from typing import cast 

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput, ImagePoseDataset
from model_interpolation_architecture import NerfModel
from model_interpolation import NerfInterpolation

MAGIC_NUMBER = 7


class CameraExtrinsics(nn.Module):
    def __init__(self, noise: float, size: int) -> None:
        super().__init__()
        
        self.noise = noise  # A noise parameter for initialization
        self.size = size    # The amount of images
        self.params = nn.Parameter(th.randn((size, 6))*noise)   # a, b, c, tx, ty, tz
        # self.register_buffer("params", th.zeros((size, 6), requires_grad=False))   # a, b, c, tx, ty, tz
        self.register_buffer("a_help_mat", th.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], requires_grad=False))
        self.register_buffer("b_help_mat", th.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], requires_grad=False))
        self.register_buffer("c_help_mat", th.tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]], requires_grad=False))


    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        # Find the appropriate parameters for the index
        # Rotation
        a = self.params[i, 0].view(-1, 1, 1)*self.a_help_mat
        b = self.params[i, 1].view(-1, 1, 1)*self.b_help_mat
        c = self.params[i, 2].view(-1, 1, 1)*self.c_help_mat

        # Translation
        trans = self.params[i, 3:] 
         
        # Create the rotation matrix
        R = th.matrix_exp(a+b+c)

        # Get the new rotation and translation
        # new_o = th.matmul(R, o.unsqueeze(-1)).squeeze(-1) + trans
        # new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)
        
        # Testing barf 
        o = th.ones_like(o)
        d = th.ones_like(d)
        new_o = th.matmul(R, o.unsqueeze(-1)).squeeze(-1) + trans
        new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)

        return new_o, new_d, trans, R
    

class NerfCameraCalibration(NerfInterpolation):
    def __init__(
        self, noise, *args, **kwargs
    ):  
        # Instansiate the nerf model as usual 
        super().__init__(*args, **kwargs)
        
        # Create camera calibration model
        self.n_training_images = len(self.trainer.datamodule.dataset_train) # type: ignore
        self.camera_extrinsics = CameraExtrinsics(noise, self.n_training_images)

    ############ pytorch lightning functions ############

    def _step_helpher(self, batch: DatasetOutput, batch_idx: int, stage: str):
        """
        general function for training and validation step
        """
        ray_origs, ray_dirs, ray_colors = batch

        # TODO: Add schedular here that changes which gaussian to use
        ray_colors = ray_colors[:, :, 0]
        
        # Compute the rgb values for the given rays for both models
        ray_colors_pred_fine, ray_colors_pred_coarse = self(ray_origs, ray_dirs)

        # Compute the individual losses
        proposal_loss = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors)
        radiance_loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors)

        # Calculate total loss
        network_loss = proposal_loss + radiance_loss
        
        # Calculate PSNR
        # NOTE: Cannot calculate SSIM because it relies on image patches
        # NOTE: Cannot calculate LPIPS because it relies on image patches
        psnr = -10 * th.log10(radiance_loss)

        # Log metrics
        self.log_dict({
            f"{stage}_network_loss": network_loss,
            f"{stage}_proposal_loss": proposal_loss,
            f"{stage}_radiance_loss": radiance_loss,
            f"{stage}_psnr": psnr,
        }) 

        return network_loss

    def training_step(self, batch: DatasetOutput, batch_idx: int):
        return self._step_helpher(batch, batch_idx, "train")

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        return self._step_helpher(batch, batch_idx, "val")


    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
        )
        scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.learning_rate_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


