from typing import Literal, Optional, cast

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from dataset import DatasetOutput
from data_module import ImagePoseDataModule
from model_camera_extrinsics import CameraExtrinsics
from model_interpolation import InnerModelBatchInput, NerfInterpolation
from model_interpolation_architecture import BarfPositionalEncoding



class CameraCalibrationModel(NerfInterpolation):
    def __init__(
        self, 
        n_training_images: int,
        camera_learning_rate: float,
        camera_learning_rate_stop_epoch: int = 10,
        camera_learning_rate_decay: float = 0.999,
        camera_learning_rate_period: float = 0.4,
        camera_weight_decay: float = 0.0,
        *inner_model_args, 
        **inner_model_kwargs,
    ):  
        super().__init__(*inner_model_args, **inner_model_kwargs)

        # self.automatic_optimization = False

        self.position_encoder = cast(BarfPositionalEncoding, self.position_encoder)
        self.direction_encoder = cast(BarfPositionalEncoding, self.direction_encoder)

        # Create camera calibration model
        self.camera_extrinsics = CameraExtrinsics(n_training_images)

        # Store hyperparameters
        self.camera_learning_rate = camera_learning_rate
        self.camera_learning_rate_stop_epoch = camera_learning_rate_stop_epoch
        self.camera_learning_rate_decay = camera_learning_rate_decay
        self.camera_learning_rate_period = camera_learning_rate_period
        self.camera_weight_decay = camera_weight_decay

        self._camera_learning_rate_milestone = camera_learning_rate_period


    # TODO: make static method.
    # TODO: change convention, such that output R has shape (1, 3, 3) - makes broadcasting easier?
    def kabsch_algorithm(self, point_cloud_from: th.Tensor, point_cloud_to: th.Tensor): 
        """
        Align "point_cloud_from" to "point_cloud_to" with a matrix R and a vector t.
        align_rotation: helper function that optimizes the subproblem ||P - R@Q||^2 using SVD


        Details:
        --------
        The output of this function is the rotation matrix R,
        the translation vector t, and the scaling factor c such that
        R@point_cloud_from*c + t estimates point_cloud_to.

        That is, the convention of this algorithm (especially importantly for R)
        is that we multiply the point cloud from the left, i.e.
        for R, t and c being the output of this function, and
        point_cloud_to_hat being the estimated point cloud
        (point_cloud_from transformed to the space of point_cloud_to)
        we have:

        point_cloud_to_hat = th.matmul(R.unsqueeze(0), point_cloud_from.unsqueeze(-1)).squeeze(-1)*c + t
                           = th.matmul(R, point_cloud_from.T).T*c + t

        
        Parameters:
        -----------
            point_cloud_from: th.Tensor(N, 3) - the point cloud where we transform from
            point_cloud_to: th.Tensor(N, 3) - the point cloud where we transform to

        Returns:
        --------
            R: th.Tensor(3, 3) - the rotation matrix
            t: th.Tensor(1, 3) - the translation vector
            c: th.Tensor(1) - the scaling factor
        """
        def align_rotation(P: th.Tensor, Q: th.Tensor):
            """
            Optimize ||P - Q@R||^2 using SVD
            """
            H = P.T@Q
            U, S, V = th.linalg.svd(H.to(dtype=th.float)) # Conventional notation is: U, S, V^T 
            d = th.linalg.det((V.T@U.T).to(dtype=th.float))
            K = th.eye(len(S), dtype=th.float, device=P.device)
            K[-1,-1] = d
            R = V.T@K@U.T
            return R.to(dtype=P.dtype)

        assert point_cloud_from.shape == point_cloud_to.shape, "point_cloud_from and point_cloud_to must have the same shape"
        assert point_cloud_from.shape[1] == 3 and len(point_cloud_from.shape) == 2, "point_cloud_from and point_cloud_to must be of shape (N, 3)"

        # Translate both point clouds to the origin 
        mean_from = th.mean(point_cloud_from, dim=0, keepdim=True)
        mean_to = th.mean(point_cloud_to, dim=0, keepdim=True)

        # Get the centered point clouds 
        point_cloud_centered_from = point_cloud_from - mean_from
        point_cloud_centered_to = point_cloud_to - mean_to

        # get scaling
        c = th.sqrt(th.sum(point_cloud_centered_to**2)) / th.sqrt(th.sum(point_cloud_centered_from**2))
        # c = th.mean(th.linalg.norm(point_cloud_centered_to, dim=1)) / th.mean(th.linalg.norm(point_cloud_centered_from, dim=1))

        # Find the rotation matrix such that ||(C1 - m1) - (C2 - m2)@R||^2 is minimized
        R = align_rotation(point_cloud_centered_from, point_cloud_centered_to)

        # The translation vector is now 
        # t = mean_to - mean_from@R

        # C1_hat = R@(C2 - m2)*c + m1 = R@c*C2 + (m1 - R@c*m2

        t = mean_to - (th.matmul(R, mean_from.T)*c).T

        return R, t, c

    
    def training_transform(self, batch: DatasetOutput) -> DatasetOutput:
        """
        Takes in a training batch and transforms the noisy rays to the predicting space
        
        Parameters:
        -----------
            batch: DatasetOutput - the batch of data to transform
            
        Returns:
        --------
            batch: DatasetOutput - the transformed batch of data
        """
        # Deconstruct batch
        (
            ray_origs_raw, 
            ray_origs_noisy, 
            ray_dirs_raw, 
            ray_dirs_noisy, 
            ray_colors_raw, 
            img_idx
        ) = batch

        # Transform rays to model prediction space
        ray_origs_pred, ray_dirs_pred, _, _ = self.camera_extrinsics(
            img_idx, 
            ray_origs_noisy, 
            ray_dirs_noisy
        )

        # Reconstruct batch
        return (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx
        )


    def validation_transform(self, batch: DatasetOutput) -> DatasetOutput:
        """
        Takes in a validation batch and transforms the noisy rays to the predicting space
        
        Parameters:
        -----------
            batch: DatasetOutput - the batch of data to transform
            
        Returns:
        --------
            batch: DatasetOutput - the transformed batch of data
        """
        # Deconstruct batch
        (
            ray_origs_raw, 
            ray_origs_noisy, 
            ray_dirs_raw, 
            ray_dirs_noisy, 
            ray_colors_raw, 
            img_idx
        ) = batch

        # Transform rays to model prediction space
        ray_origs_pred, ray_dirs_pred, _ = self.validation_transform_rays(
            ray_origs_raw,
            ray_dirs_raw
        )

        # Reconstruct batch
        return (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx
        )


    def compute_post_transform_params(
            self,
            ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Takes in validation rays and transforms them to the predicting space
        And predicts the post-transform parameters (R, t, c) to use
        """

        # Get the raw and noise origins 
        # origs raw: the true origins of the rays
        # origs noisy: the intial guess of the origins of the rays 
        #               (provided by datamodule and fed to camera extrinsics)
        dataset = cast(ImagePoseDataModule, self.trainer.datamodule).dataset_train
        # NOTE (lauge): Get the original indices of the images - to be passed to camera_extrinsics
        img_idxs = list(dataset.index_to_index.values())
        img_idxs = th.tensor(img_idxs, device=self.device, dtype=th.int32)
        origs_raw = dataset.camera_origins.to(self.device)
        origs_noisy = dataset.camera_origins_noisy.to(self.device)
        origs_pred, _ = self.camera_extrinsics.forward_origins(img_idxs, origs_noisy)


        # Align raw space to predicted model space
        post_transform_params = self.kabsch_algorithm(origs_raw, origs_pred)

        return post_transform_params


    def validation_transform_rays(self,
                                  origs_val: th.Tensor,
                                  dirs_val: th.Tensor,
                                  post_transform_params: Optional[tuple[th.Tensor, th.Tensor]] = None,
                                  ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Takes in validation rays and transforms them to the predicting space 

        Parameters:
        -----------
            origs_val: th.Tensor - the origins of the rays
            dirs_val: th.Tensor - the directions of the rays
            post_transform_params: Optional[tuple[th.Tensor, th.Tensor]] - the post-transform parameters (R, t) to use. If None, then they are calculated

        Returns:
        --------
            origs_model: th.Tensor - the new origins of the rays
            dirs_model: th.Tensor - the new directions of the rays
            translations: th.Tensor - the translation vectors
            rotations: th.Tensor - the rotation matrices
        """
        # return origs_val, dirs_val, None
        # If not supplied, get the rotation matrix and the translation vector 
        if post_transform_params is None:
            post_transform_params = self.compute_post_transform_params()

        # Deconstruct post-transform parameters
        R, t, c = post_transform_params

        # Transform the validation image to this space 
        origs_model = th.matmul(R, origs_val.unsqueeze(-1)).squeeze(-1)*c + t
        dirs_model = th.matmul(R, dirs_val.unsqueeze(-1)).squeeze(-1)

        return origs_model, dirs_model, post_transform_params   


    def training_step(self, batch, batch_idx):

        n_batches = len(self.trainer.train_dataloader)
        epoch = self.trainer.current_epoch + batch_idx/n_batches

        self.position_encoder.update_alpha(epoch)
        self.direction_encoder.update_alpha(epoch)

        return self._step_helper(batch, batch_idx, "train")


    def validation_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, "val")

    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        for value in batch:
            assert not th.isnan(value).any(), "NaN values in raw batch"
        # Transform batch to model prediction space
        if purpose == "train": batch = self.training_transform(batch)
        elif purpose == "val": batch = self.validation_transform(batch)

        for value in batch:
            assert not th.isnan(value).any(), "NaN values in transformed batch"
        
        # unpack batch
        (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx
        ) = batch

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self(ray_origs_pred, ray_dirs_pred)

        assert not th.isnan(ray_colors_pred_fine).any(), "NaN values in ray_colors_pred_fine"
        assert not th.isnan(ray_colors_pred_coarse).any(), "NaN values in ray_colors_pred_coarse"

        # compute the loss
        loss_fine = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw)
        loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw)

        psnr = -10 * th.log10(loss_fine)

        # Log metrics
        self.log_dict({f"{purpose}_loss_fine": loss_fine,
                       f"{purpose}_loss_coarse": loss_coarse,
                       f"{purpose}_psnr": psnr})
        
        return loss_fine + loss_coarse

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            [
                {"params": self.model_coarse.parameters(), "lr": self.learning_rate},
                {"params": self.model_fine.parameters(), "lr": self.learning_rate},
                {"params": self.camera_extrinsics.parameters(), "lr": self.camera_learning_rate}
            ]
        )
        return optimizer