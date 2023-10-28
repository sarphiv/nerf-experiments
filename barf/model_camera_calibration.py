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
    
    def get_rotations(self, i: th.Tensor) -> th.Tensor:
        """
        Get the rotation matrices for the given indexs

        Parameters:
        -----------
            i: th.Tensor - the indexs of the rotation matrices to get

        Returns:
        --------
            R: th.Tensor - the rotation matrices
        """
        # Helpers
        a = self.params[i, 0].view(-1, 1, 1)*self.a_help_mat
        b = self.params[i, 1].view(-1, 1, 1)*self.b_help_mat
        c = self.params[i, 2].view(-1, 1, 1)*self.c_help_mat

        # Rotation matrix 
        R = th.matrix_exp(a+b+c)

        return R

    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forwards pass: Gets the origins and directions in the predicted camera space
        """
        # Translation
        trans = self.params[i, 3:] 
         
        # Create the rotation matrix
        R = self.get_rotations(i)

        # Get the new rotation and translation
        new_o = th.matmul(R, o.unsqueeze(-1)).squeeze(-1) + trans
        new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)

        return new_o, new_d, trans, R

class NerfCameraCalibration(NerfInterpolation):
    def __init__(
        self, noise: float, *args, **kwargs
    ):  
        # Instansiate the nerf model as usual 
        super().__init__(*args, **kwargs)
        
        # Create camera calibration model - here we need a seperate index for each image
        self.n_training_images = len(self.trainer.datamodule.dataset_train) # type: ignore
        self.camera_extrinsics = CameraExtrinsics(noise, self.n_training_images)
    
    
    def kabsch_algorithm(self, point_cloud_1: th.Tensor, point_cloud_2: th.Tensor): 
        """
        Align "point_cloud_2" to "point_cloud_1" with a matrix R and a vector t.
        align_rotation: helper function that optimizes the subproblem ||P - Q@R||^2 using SVD
        
        Parameters:
        -----------
            point_cloud_1: th.Tensor - the point cloud where we transform to
            point_cloud_2: th.Tensor - the point cloud where we transform from

        Returns:
        --------
            R: th.Tensor - the rotation matrix
            t: th.Tensor - the translation vector
        """
        def align_rotation(P, Q):
            """
            Optimize ||P - Q@R||^2 using SVD
            """
            H = P.T@Q
            U, S, V = th.linalg.svd(H) # In normal notation this is U, S, V^T 
            d = th.linalg.det(V.T@U.T)
            K = th.eye(len(S))
            K[-1,-1] = d
            R = V.T@K@U.T
            return R

        # Translate both point clouds to the origin 
        mean_1 = th.mean(point_cloud_1, dim=0, keepdim=True)
        mean_2 = th.mean(point_cloud_2, dim=0, keepdim=True)

        # Find the rotation matrix such that ||(C1 - m1) - (C2 - m2)@R||^2 is minimized
        R = align_rotation(point_cloud_1 - mean_1, point_cloud_2 - mean_2)

        # The translation vector is now 
        t = mean_1 - mean_2@R

        return R, t
    
    def validation_transform(self, val_origs: th.Tensor, val_dirs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Takes in validation rays and transforms them to the predicting space 

        Parameters:
        -----------
            val_origs: th.Tensor - the origins of the rays
            val_dirs: th.Tensor - the directions of the rays

        Returns:
        --------
            new_origs: th.Tensor - the new origins of the rays
            new_dirs: th.Tensor - the new directions of the rays
            translations: th.Tensor - the translation vectors
            rotations: th.Tensor - the rotation matrices
        """
        # Get the raw and noise origins 
        raw_origs = self.trainer.datamodule.dataset_train.raw_origins # type: ignore
        noise_origs = self.trainer.datamodule.dataset_train.noise_origins # type: ignore
        indexes = th.arange(raw_origs.shape[0]) #TODO: check that this works and corresponds to [i[-1] for i in self.trainer.datamodule.dataset_train.dataset] ish 

        # Get the predicted origins 
        pred_origs = self.camera_extrinsics(noise_origs, indexes)

        #TODO: add a flag so that we do not need to calculate the rotation matrix and translation vector every time if the network has not changed 
        # Get the rotation matrix and the translation vector 
        R, t = self.kabsch_algorithm(self, pred_origs, raw_origs) # type: ignore

        # Transform the validation image to this space 
        new_origs = th.matmul(R, val_origs.unsqueeze(-1)).squeeze(-1) + t
        new_dirs = th.matmul(R, val_dirs.unsqueeze(-1)).squeeze(-1)

        return new_origs, new_dirs, t, R
    
    
    ############ pytorch lightning functions ############

    def _step_helpher(self, ray_origs, ray_dirs, ray_colors, stage: str):
        """
        general function for training and validation step
        """
        # Call the forward pass 
        ray_colors_pred_coarse, ray_colors_pred_fine = self.forward(ray_origs, ray_dirs)

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
        # Get the rays to sample from
        ray_origs, ray_dirs, ray_colors, ray_index = batch

        # Transform to the prediction space 
        ray_origs, ray_dirs, translations, rotations = self.camera_extrinsics(ray_index, ray_origs, ray_dirs)

        # TODO: Add schedular here that changes which gaussian to use (for now -1 one should be the last)
        ray_colors = ray_colors[:, :, -1]

        return self._step_helpher(ray_origs, ray_dirs, ray_colors, "train")

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        # Get the rays to sample from
        ray_origs, ray_dirs, ray_colors, _ = batch

        # Transform to the predicted coordinates 
        ray_origs, ray_dirs, translations, rotations = self.validation_transform(ray_origs, ray_dirs)

        # Here we only care about the ray that has no noise applied to it, hence the -1
        ray_colors = ray_colors[:, :, -1]

        return self._step_helpher(ray_origs, ray_dirs, ray_colors, "train")


    def configure_optimizers(self):
        # TODO: Call super and set up optimizers and scheduler for this specific class
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


