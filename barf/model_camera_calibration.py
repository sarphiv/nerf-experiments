from typing import cast 

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput, ImagePoseDataset
from model_interpolation_architecture import NerfModel
from model_interpolation import NerfInterpolation

MAGIC_NUMBER = 7


class CameraExtrinsics(nn.Module):
    #TODO: Instead of learning the identity, define a new rotation matrix that is applied before the learned one
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
    
    def get_training_rotation(self, indexes: th.Tensor, origins: th.Tensor) -> th.Tensor:
        """
        Finds the rotation matrix that minimizes the point clouds for origins and R origins. 

        Parameters:
        -----------
            indexs: th.Tensor - the indexs all training images
            origins: th.Tensor - the origins of the rays/images

        Returns:
        --------
            T: th.Tensor - the translation vector
            R: th.Tensor - the rotation matrix
        """
        # Extrinsics oriontation
        R = self.get_rotations(indexes)

        # Get the translations 
        trans = self.params[indexes, 3:] 

        # Get the predicted extrinsics origins
        o = th.matmul(R, origins.unsqueeze(-1)).squeeze(-1) + trans 

        def align_rotation(P, Q):
            """
            optimize ||P@R - Q||^2 using SVD
            """
            H = P.T@Q
            U, S, V = th.linalg.svd(H)
            d = th.linalg.det(V@U.T)
            K = th.eye(len(S))
            K[-1,-1] = d
            R = U@K@V.T
            return R

        def align_paired_point_clouds(P, Q):
            """
            align paired point clouds P and Q
            by translating and rotating P into Q
            """
            # translate P to origin
            cP = th.mean(P, dim=0, keepdim=True)
            cQ = th.mean(Q, dim=0, keepdim=True)
            # rotate P to Q
            R = align_rotation(P-cP, Q-cQ)
            Qhat = (P - cP)@R + cQ
            return Qhat, R, cQ-cP@R # t = cQ@R - cP

        raise NotImplementedError("This function is not implemented yet")
    
    

class NerfCameraCalibration(NerfInterpolation):
    def __init__(
        self, noise: float, *args, **kwargs
    ):  
        # Instansiate the nerf model as usual 
        super().__init__(*args, **kwargs)
        
        # Create camera calibration model - here we need a seperate index for each image
        self.n_training_images = len(self.trainer.datamodule.dataset_train) # type: ignore
        self.camera_extrinsics = CameraExtrinsics(noise, self.n_training_images)
    
    # def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor, ray_index: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    #     """
    #     First apply the camera extrinsics, then the nerf as in vanilla
    #     """
    #     # Get the new origins and directions with the camera extrinsics module
    #     ray_origs, ray_dirs, translations, rotations = self.camera_extrinsics(ray_index, ray_origs, ray_dirs)

    #     # Run the vanilla nerf model 
    #     return super().forward(ray_origs, ray_dirs)
    
    def kabsc(self, raw_origs: th.Tensor, noise_origs: th.Tensor, indexes: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Find the best rotation and translation to map the given camera poses to the raw (true) ones. 
        """
        pred_origs = self.camera_extrinsics(noise_origs, indexes)

        return kabsc(pred_origs, raw_origs) -> R, T
        raise NotImplementedError("This function is not implemented yet")
    
    def validation_transform() -> ray_origs, ray_dirs
        self.trainer.datamodule._get_dataset("train")
        self.trainer.datamodule.dataset_train.raw_origins
        self.trainer.datamodule.dataset_train.noise_origins
        self.trainer.datamodule.train_dataloader()
        raise NotImplementedError()
    

    # def validation_suck_ass(self, ray_origs: th.Tensor, ray_dirs: th.Tensor, ray_index: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    #     """
    #     Forward pass of the model for a validation step. 
    #     When we are setting the camera extrinsics, the validation rays must be taken to the correct space.
    #     """
    #     #TODO: add a flag so that we do not need to calculate the rotation matrix and translation vector every time if the network has not changed 
    #     # Calculate the rotation matrix and translation vector
    #     translation, rotation = self.camera_extrinsics.get_training_rotation(ray_index, ray_origs, ray_dirs)

    #     # Transform the validation image to this space 
    #     new_origs = th.matmul(rotation, ray_origs.unsqueeze(-1)).squeeze(-1) + translation
    #     new_dirs = th.matmul(rotation, ray_dirs.unsqueeze(-1)).squeeze(-1)

    #     # Run the vanilla nerf model 
    #     return super().forward(new_origs, new_dirs)
    
    ############ pytorch lightning functions ############

    def _step_helpher(self, batch: DatasetOutput, stage: str):
        """
        general function for training and validation step
        """
        # TODO: Move these two lines to the respective train and val 
        # Get the rays to sample from
        ray_origs, ray_dirs, ray_colors, ray_index = batch
        # TODO: Add schedular here that changes which gaussian to use
        ray_colors = ray_colors[:, :, 0]

        # Run forwards to get color predictions
        if stage == "train":
            ray_colors_pred_fine, ray_colors_pred_coarse = self(ray_origs, ray_dirs, ray_index)
        else: 
            ray_colors_pred_fine, ray_colors_pred_coarse = self.validation_forward(ray_origs, ray_dirs, ray_index)
        
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
        # TODO: Do this here (fix later)
        ray_origs, ray_dirs, translations, rotations = self.camera_extrinsics(ray_index, ray_origs, ray_dirs)
        return self._step_helpher(batch, "train")

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        # TODO: make this later (validation transform) 
        return self._step_helpher(batch, "val")


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


