from typing import Literal, Optional, cast

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from dataset import DatasetOutput
from data_module import ImagePoseDataModule
from model_camera_extrinsics import CameraExtrinsics
from model_interpolation import NerfInterpolation



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

        # Create camera calibration model
        self.camera_extrinsics = CameraExtrinsics(n_training_images)

        # Store hyperparameters
        self.camera_learning_rate = camera_learning_rate
        self.camera_learning_rate_stop_epoch = camera_learning_rate_stop_epoch
        self.camera_learning_rate_decay = camera_learning_rate_decay
        self.camera_learning_rate_period = camera_learning_rate_period
        self.camera_weight_decay = camera_weight_decay

        self._camera_learning_rate_milestone = camera_learning_rate_period



    def kabsch_algorithm(self, point_cloud_from: th.Tensor, point_cloud_to: th.Tensor): 
        """
        Align "point_cloud_from" to "point_cloud_to" with a matrix R and a vector t.
        align_rotation: helper function that optimizes the subproblem ||P - Q@R||^2 using SVD
        
        Parameters:
        -----------
            point_cloud_from: th.Tensor - the point cloud where we transform from
            point_cloud_to: th.Tensor - the point cloud where we transform to

        Returns:
        --------
            R: th.Tensor - the rotation matrix
            t: th.Tensor - the translation vector
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

        # Translate both point clouds to the origin 
        mean_from = th.mean(point_cloud_from, dim=0, keepdim=True)
        mean_to = th.mean(point_cloud_to, dim=0, keepdim=True)

        # Find the rotation matrix such that ||(C1 - m1) - (C2 - m2)@R||^2 is minimized
        R = align_rotation(point_cloud_to - mean_to, point_cloud_from - mean_from)

        # The translation vector is now 
        t = mean_to - mean_from@R

        return R, t
    
    
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
            ray_colors_blur, 
            ray_scales, 
            img_idx
        ) = batch

        # Transform rays to model prediction space
        ray_origs_noisy, ray_dirs_noisy, _, _ = self.camera_extrinsics(
            img_idx, 
            ray_origs_noisy, 
            ray_dirs_noisy
        )

        # Reconstruct batch
        return (
            ray_origs_raw, 
            ray_origs_noisy, 
            ray_dirs_raw, 
            ray_dirs_noisy, 
            ray_colors_raw, 
            ray_colors_blur, 
            ray_scales, 
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
            ray_colors_blur, 
            ray_scales, 
            img_idx
        ) = batch

        # Transform rays to model prediction space
        ray_origs_noisy, ray_dirs_noisy, _ = self.validation_transform_rays(
            ray_origs_noisy, 
            ray_dirs_noisy
        )

        # Reconstruct batch
        return (
            ray_origs_raw, 
            ray_origs_noisy, 
            ray_dirs_raw, 
            ray_dirs_noisy, 
            ray_colors_raw, 
            ray_colors_blur, 
            ray_scales, 
            img_idx
        )


    def validation_transform_rays(self, origs_val: th.Tensor, dirs_val: th.Tensor, post_transform_params: Optional[tuple[th.Tensor, th.Tensor]] = None) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, th.Tensor]]:
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

        # If not supplied, get the rotation matrix and the translation vector 
        if post_transform_params is None:
            # Get the raw and noise origins 
            (origs_raw, _), (origs_noisy, _) = cast(ImagePoseDataModule, self.trainer.datamodule).train_camera_center_rays(self.device) # type: ignore
            # origs_raw = th.cat([origs[0, 0] for origs in self.trainer.datamodule.dataset_train.origins_raw.values()]).to(device=origs_val.device).view(-1, 3) # type: ignore
            # origs_noisy = th.cat([origs[0, 0] for origs in self.trainer.datamodule.dataset_train.origins_noisy.values()]).to(device=origs_val.device).view(-1, 3) # type: ignore
            img_idxs = th.arange(len(origs_raw), device=origs_raw.device)

            # Get the predicted origins 
            origs_pred, _, _ = self.camera_extrinsics.forward_origins(img_idxs, origs_noisy)

            # Align raw space to predicted model space
            post_transform_params = self.kabsch_algorithm(origs_raw, origs_pred)


        # Deconstruct post-transform parameters
        R, t = post_transform_params

        # Transform the validation image to this space 
        origs_model = th.matmul(R, origs_val.unsqueeze(-1)).squeeze(-1) + t
        dirs_model = th.matmul(R, dirs_val.unsqueeze(-1)).squeeze(-1)

        return origs_model, dirs_model, (R, t)




    def _camera_optimizer_step(self, loss: th.Tensor):
        # NOTE: Assuming camera loss is calculated right after radiance loss.
        #  This saves a backwards pass by using the same graph.
        #  If this is not the case, then the step function should be altered
        self._camera_optimizer.zero_grad()
        self.manual_backward(loss)
        self._camera_optimizer.step()


    def _camera_scheduler_step(self, batch_idx: int):
        epoch_fraction = self.trainer.current_epoch + batch_idx/self.trainer.num_training_batches

        if (
            epoch_fraction >= self._camera_learning_rate_milestone and
            epoch_fraction <= self.camera_learning_rate_stop_epoch
        ):
            self._camera_learning_rate_milestone += self.camera_learning_rate_period
            self._camera_learning_rate_scheduler.step()


    def _get_logging_losses(
        self, 
        stage: Literal["train", "val", "test"], 
        proposal_loss: th.Tensor,
        radiance_loss: th.Tensor, 
        radiance_loss_raw: th.Tensor,
        camera_loss: th.Tensor,
        *args, 
        **kwargs
    ) -> dict[str, th.Tensor]:
        losses = super()._get_logging_losses(
            stage,
            proposal_loss,
            radiance_loss,
            *args,
            **kwargs
        )

        # Calculate PSNR
        # NOTE: Cannot calculate SSIM because it relies on image patches
        # NOTE: Cannot calculate LPIPS because it relies on image patches
        psnr = -10 * th.log10(radiance_loss_raw)

        return {
            **losses,
            f"{stage}_radiance_loss_raw": radiance_loss_raw,
            f"{stage}_psnr_raw": psnr,
        }




    def _forward_loss(self, batch: DatasetOutput):
        # Decontsruct batch
        (
            ray_origs_raw, 
            ray_origs_noisy, 
            ray_dirs_raw, 
            ray_dirs_noisy, 
            ray_colors_raw, 
            ray_colors_blur, 
            ray_scales, 
            img_idx
        ) = batch

        # Forward on model
        ray_colors_pred, (proposal_loss_blur, radiance_loss_blur) = super()._forward_loss((
            ray_origs_noisy, 
            ray_dirs_noisy, 
            ray_colors_blur, 
            ray_scales
        ))


        # Calculate raw radiance loss based on real color
        radiance_loss_raw = nn.functional.mse_loss(ray_colors_pred, ray_colors_raw)

        # Compute the camera loss
        # TODO: Should probably be changed to a different loss function
        camera_loss = nn.functional.mse_loss(ray_colors_pred, ray_colors_blur)


        # Return color prediction and losses
        return (
            ray_colors_pred,
            (proposal_loss_blur, radiance_loss_blur, radiance_loss_raw, camera_loss)
        )



    def training_step(self, batch: DatasetOutput, batch_idx: int):
        """Perform a single training forward pass, optimization step, and logging.
        
        Args:
            batch (InnerModelBatchInput): Batch of data
            batch_idx (int): Index of the batch
        
        Returns:
            th.Tensor: Loss
        """
        # Transform batch to model prediction space
        batch = self.training_transform(batch)

        # Forward pass
        _, (proposal_loss_blur, radiance_loss_blur, radiance_loss_raw, camera_loss) = self._forward_loss(batch)

        # Backward pass and step through each optimizer
        self._proposal_optimizer_step(proposal_loss_blur)
        self._radiance_optimizer_step(radiance_loss_blur)
        # NOTE: Assuming camera loss is calculated right after radiance loss.
        #  This saves a backwards pass by using the same graph.
        #  If this is not the case, then the step function should be altered
        self._camera_optimizer_step(camera_loss)

        # Step learning rate schedulers
        self._proposal_scheduler_step(batch_idx)
        self._radiance_scheduler_step(batch_idx)
        self._camera_scheduler_step(batch_idx)

        # Log metrics
        self.log_dict(self._get_logging_losses(
            "train",
            proposal_loss_blur,
            radiance_loss_blur,
            radiance_loss_raw,
            camera_loss
        ))


        # Return loss
        return radiance_loss_raw


    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        # Transform to the model prediction space
        # TODO: Cache post-transformation params from Kabsc algorithm
        batch = self.validation_transform(batch)

        # Forward pass
        _, (proposal_loss_blur, radiance_loss_blur, radiance_loss_raw, camera_loss) = self._forward_loss(batch)

        # Log metrics
        self.log_dict(self._get_logging_losses(
            "val",
            proposal_loss_blur,
            radiance_loss_blur,
            radiance_loss_raw,
            camera_loss
        ))

        return radiance_loss_raw




    def configure_optimizers(self):
        # Configure super optimizers
        optimizers, schedulers = super().configure_optimizers()
        
        # Set up optimizer and schedulers for camera extrinsics
        self._camera_optimizer = th.optim.Adam(
            self.camera_extrinsics.parameters(), 
            lr=self.camera_learning_rate, 
            weight_decay=self.camera_weight_decay,
        )

        self._camera_learning_rate_scheduler = th.optim.lr_scheduler.ExponentialLR(
            self._camera_optimizer, 
            gamma=self.camera_learning_rate_decay
        )


        # Set optimizers and schedulers
        return (
            optimizers + [self._camera_optimizer],
            schedulers + [self._camera_learning_rate_scheduler]
        )


