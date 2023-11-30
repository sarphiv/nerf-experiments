from typing import Literal, Optional, cast
import warnings

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.optimizer import Optimizer

from dataset import DatasetOutput
from data_module import ImagePoseDataModule
from model_camera_extrinsics import CameraExtrinsics
from model_interpolation import InnerModelBatchInput, NerfInterpolation
from model_interpolation_architecture import BarfPositionalEncoding



class CameraCalibrationModel(NerfInterpolation):
    def __init__(
        self, 
        n_training_images: int,
        camera_learning_rate_start: float,
        camera_learning_rate_stop: float,
        camera_learning_rate_stop_step: int = -1,
        camera_weight_decay: float = 0.0,
        max_gaussian_sigma: float = 0.0,
        *inner_model_args, 
        **inner_model_kwargs,
    ):  
        super().__init__(*inner_model_args, **inner_model_kwargs)



        self.position_encoder = cast(BarfPositionalEncoding, self.position_encoder)
        self.direction_encoder = cast(BarfPositionalEncoding, self.direction_encoder)

        # Create camera calibration model
        self.camera_extrinsics = CameraExtrinsics(n_training_images)

        # Store hyperparameters
        self.camera_learning_rate_start = camera_learning_rate_start
        self.camera_learning_rate_stop = camera_learning_rate_stop
        self.camera_learning_rate_stop_step = camera_learning_rate_stop_step

        self.camera_weight_decay = camera_weight_decay

        self.max_gaussian_sigma = max_gaussian_sigma

    ##############################################################
    # Batch transformation helpers

    # TODO: make static method.
    # TODO: maybe change convention, such that output R has shape (1, 3, 3) - makes broadcasting easier?
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

        # C1_hat = R@(C2 - m2)*c + m1 = R@c*C2 + (m1 - R@c*m2) => t = m1 - R@c*m2
        t = mean_to - (th.matmul(R, mean_from.T)*c).T

        return R, t, c


    def validation_transform_rays(self,
                                  origs_val: th.Tensor,
                                  dirs_val: th.Tensor,
                                  post_transform_params: Optional[tuple[th.Tensor, th.Tensor]] = None,
                                  ) -> tuple[th.Tensor, th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """
        Takes in validation rays and transforms them to the predicting space 
        using the post_transform_params computed in compute_post_transform_params()

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


    def compute_post_transform_params(
            self,
            from_raw_to_pred: bool = True,
            return_origs: bool = False,
            ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Computes the transform params (R, t, c) used for transforming
        the validation rays to the predicting space. 

        It relies on the fact that the ground truth camera origins are known,
        but essentially (ideally) not used in training.

        Works by using the Kabsch algorithm to transform ground truth origins
        to the predicted origins (as predicted by the camera_extrinsics model).

        It outputs the rotation matrix R, the translation vector t, and the
        scaling factor c such that R@origs_true_train*c + t estimates origs_pred_train.

        These can then be used to estimate the origins of the validation rays,
        (but this is not done in this method).

        Takes no parameters, as it uses the dataset_train of the datamodule, and the camera_extrinsics
        model to get the raw and predicted train origins.

        Returns:
        --------
            R: th.Tensor(3, 3) - the rotation matrix
            t: th.Tensor(1, 3) - the translation vector
            c: th.Tensor(1) - the scaling factor
        """

        # Get the raw and noise origins 
        # origs raw: the true origins of the rays
        # origs noisy: the initial guess of the origins of the rays 
        #               (provided by datamodule and fed to camera extrinsics)
        dataset = cast(ImagePoseDataModule, self.trainer.datamodule).dataset_train
        # NOTE (lauge): Get the original indices of the images - to be passed to camera_extrinsics
        img_idxs = list(dataset.index_to_index.values())
        img_idxs = th.tensor(img_idxs, device=self.device, dtype=th.int32)
        origs_raw = dataset.camera_origins.to(self.device)
        origs_noisy = dataset.camera_origins_noisy.to(self.device)
        origs_pred, _ = self.camera_extrinsics.forward_origins(img_idxs, origs_noisy)

        # Align raw space to predicted model space
        if from_raw_to_pred:
            post_transform_params = self.kabsch_algorithm(origs_raw, origs_pred)
        else: 
            post_transform_params = self.kabsch_algorithm(origs_pred, origs_raw)
        
        # For computing pose errors
        if return_origs:
            return post_transform_params, origs_raw, origs_pred
        return post_transform_params

    ##############################################################
    # Batch transformations

    def validation_transform(self, batch: DatasetOutput) -> DatasetOutput:
        """
        Takes in a validation batch and transforms the noisy rays to the predicting space
        using the validation_transform_rays() method.
        
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

    
    def training_transform(self, batch: DatasetOutput) -> DatasetOutput:
        """
        Takes in a training batch and transforms the noisy rays (which
        are essentially the initial guesses of the model) to the
        predicting space using the self.camera_extrinsics model.
        
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


    def get_blurred_pixel_colors(self, batch: InnerModelBatchInput, sigmas: list[float], sigma: float) -> InnerModelBatchInput:
        """
        Compute the interpolation of the blurred pixel colors
        to get the blurred pixel color used for training, while also
        outputting the original pixel color.

        Parameters:
        -----------
            batch: InnerModelBatchInput - the batch of data to transform
            one of the elements is ray_colors of shape (N, n_sigmas, 3) 
            sigmas: list[float] - the sigmas used for the gaussian blur
            sigma: float - the sigma to use for the interpolation
        
        Returns:
        -----------
            batch: InnerModelBatchInput - the transformed batch of data
            now ray_colors is of shape (N, 2, 3)

        
        Details:
        --------

        The batch contains 6 different elements.
        One of the elements contains the ray_colors (this is
        what it is called in _step_helper() etc.).
        ray_colors is a Tensor of shape (N, n_sigmas, 3), where
        N is the number of rays in the batch, and n_sigmas is the number
        of sigmas used for the blurring. 

        This method only modifies the ray_colors element of the batch.
        It is modified to have shape (N, 2, 3), where 
        ray_colors[:, 0, :] are the original pixel colors, and
        ray_colors[:, 1, :] are the blurred pixel colors.

        
        """
        # Unpack batch
        (
            ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            ray_colors_raw,
            img_idx
        ) = batch

        # Find the sigma closest to the given sigma
        index_low = 0
        for index_high, s in enumerate(sigmas):
            if s < sigma: break
            index_low = index_high  
        
        # ls_1 + (1-l)s_2 = s <=> l = (s - s_2) / (s_1 - s_2)
        interpolation_coefficient = (sigma - sigmas[index_high]) / (sigmas[index_low] - sigmas[index_high] + 1e-8)
        
        # Make interpolation 
        interpolation = ray_colors_raw[:,index_low] * (interpolation_coefficient) + ray_colors_raw[:,index_high] * (1-interpolation_coefficient)
        
        return (ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            th.stack([interpolation, ray_colors_raw[:,-1]], dim=1),
            img_idx)
    
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


    def compute_pose_error(self):
        """
        Helper function that computes the pose error distances
        """
        # Get transformation from pred to raw space
        (R, t, c), origs_raw, origs_pred = self.compute_post_transform_params(from_raw_to_pred=False, return_origs=True)
        # Transform preds to raw 
        origs_pred_aligned = th.matmul(R, origs_pred.unsqueeze(2)).squeeze(2)*c + t
        # Compute error
        error = (((origs_raw - origs_pred_aligned)**2).sum(dim=1)**0.5).mean()
        return error
    

    ##############################################################
    # helper methods for the lightning module methods

    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        # Transform batch to model prediction space
        if purpose == "train":
            batch = self.training_transform(batch)
            
            # Update alpha (for positional encoding in BARF)
            n_batches = len(self.trainer.train_dataloader)
            epoch = self.trainer.current_epoch + batch_idx/n_batches

            self.position_encoder.update_alpha(epoch)
            self.direction_encoder.update_alpha(epoch)
            
        elif purpose == "val": batch = self.validation_transform(batch)

        # interpolate the blurred pixel colors
        sigma = CameraCalibrationModel.get_sigma_alpha(self.position_encoder.alpha, self.max_gaussian_sigma)
        batch = self.get_blurred_pixel_colors(batch, self.trainer.datamodule.gaussian_blur_sigmas, sigma) 


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
        loss_fine = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:,-1])
        psnr = -10 * th.log10(loss_fine)

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:,-1]) #TODO fix interpolation
            loss = loss_fine + loss_coarse

            # Log metrics
            log_dict = {f"{purpose}_loss_fine": loss_fine,
                        f"{purpose}_loss_coarse": loss_coarse,
                        f"{purpose}_psnr": psnr,
                        "alpha": self.position_encoder.alpha.item(),
                        "sigma": sigma.item(),
                        }
        else:
            loss = loss_fine
            log_dict = {f"{purpose}_loss_fine": loss_fine,
                        f"{purpose}_psnr": psnr,
                        "alpha": self.position_encoder.alpha.item(),
                        "sigma": sigma.item(),
                        }
            
        if purpose == "val" and batch_idx == 0:
            # Compute pose error
            pose_error = self.compute_pose_error()
            log_dict[f"{purpose}_pose_error"] = pose_error

        # Log metrics
        self.log_dict(log_dict)
        
        return loss


    ##############################################################
    # Lightning Module Methods


    def training_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, "train")


    def validation_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, "val")


    def configure_optimizers(self):
        # Create optimizer and schedular 
        if self.proposal:

            optimizer = th.optim.Adam(
                [
                    {"params": self.model_coarse.parameters(), "lr": self.learning_rate_start},
                    {"params": self.model_fine.parameters(), "lr": self.learning_rate_start},
                    {"params": self.camera_extrinsics.parameters(), "lr": self.camera_learning_rate_start}
                ]
            )

            lr_scheduler = SchedulerLeNice(
                optimizer, 
                start_LR=[self.learning_rate_start, self.learning_rate_start, self.camera_learning_rate_start], 
                stop_LR= [self.learning_rate_stop,  self.learning_rate_stop,  self.camera_learning_rate_stop], 
                number_of_steps=[self.learning_rate_stop_step, self.learning_rate_stop_step, self.camera_learning_rate_stop_step],
                verbose=False
            )
        else:
            optimizer = th.optim.Adam(
                [
                    {"params": self.model_coarse.parameters(), "lr": self.learning_rate_start},
                    {"params": self.camera_extrinsics.parameters(), "lr": self.learning_rate_start},
                ]
            )
            lr_scheduler = SchedulerLeNice(
                optimizer, 
                start_LR=[self.learning_rate_start, self.camera_learning_rate_start], 
                stop_LR= [self.learning_rate_stop,  self.camera_learning_rate_stop], 
                number_of_steps=[self.learning_rate_stop_step, self.camera_learning_rate_stop_step],
                verbose=False
            )
        

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "le_nice_lr_scheduler",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
class SchedulerLeNice(th.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: th.optim.Optimizer, start_LR: list[float], stop_LR: list[float], number_of_steps: list[float], verbose=False) -> None:
        # Store extra parameters 
        self.start_LR = start_LR
        self.stop_LR = stop_LR
        self.number_of_steps = number_of_steps

        # Calculate decay factors
        # Solve: start : s, end : e, decay : d, number of epochs : n 
        # s * d^n = e <=> d = (e/s)^(1/n)
        self.decay_factors = []
        for i, _ in enumerate(optimizer.param_groups):
            self.decay_factors.append((self.stop_LR[i] / (self.start_LR[i] + 1e-12)) ** (1 / self.number_of_steps[i]))

        super().__init__(optimizer,verbose=verbose)
        


    
    def get_lr(self):
        # Original function 
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        # Update lr for each individually 
        return [(group['lr'] * self.decay_factors[i] if self._step_count < self.number_of_steps[i] else group["lr"]) for i, group in enumerate(self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [base_lr * self.decay_factors[i] ** max(self._step_count, self.number_of_steps[i]) for i, base_lr in enumerate(self.start_LR)]