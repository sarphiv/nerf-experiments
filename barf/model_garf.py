from typing import Callable, Literal, Optional, Dict

import torch as th
import torch.nn.functional as F
import pytorch_lightning as pl
import nerfacc

from model_garf_radiance import RadianceNetwork
from model_garf_proposal import ProposalNetwork
from model_camera_calibration import CameraCalibrationModel
from model_camera_extrinsics import CameraExtrinsics


# Type alias for inner model batch input
#  (origin, direction, pixel_color, pixel_relative_blur)
InnerModelBatchInput = tuple[th.Tensor, th.Tensor, th.Tensor]


class GarfModel(CameraCalibrationModel):
    def __init__(
        self, 
        n_train_images: int = 100,
        camera_learning_rate_start: float = 1e-4,
        camera_learning_rate_stop: float = 1e-5,
        camera_learning_rate_decay_end: int = 10000,
        near_plane: float = 2,
        far_plane: float =7,
        proposal_samples_per_ray: int = 64,
        radiance_samples_per_ray: int = 192,
        gaussian_init_min: float = 0.0,
        gaussian_init_max: float = 1.0,
        gaussian_learning_rate_factor: float = 1.0,
        proposal_learning_rate_start: float = 1e-4,
        proposal_learning_rate_stop: float = 1e-5,
        proposal_learning_rate_decay_end: int = 10000, #step
        proposal_weight_decay: float = 0.0,
        radiance_learning_rate_start: float = 1e-4,
        radiance_learning_rate_stop: float = 1e-5,
        radiance_learning_rate_decay_end: int = 10000, #step
        radiance_weight_decay: float = 0.0,
    ):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters()


        # The near and far sphere distances 
        self.near_sphere_normalized = near_plane
        self.far_sphere_normalized = far_plane

        # Samples per ray for proposal network for the density estimation
        self.proposal_samples_per_ray = proposal_samples_per_ray
        # Samples per ray for radiance network for the volumetric rendering
        self.radiance_samples_per_ray = radiance_samples_per_ray
        


        # Proposal network estimates sampling density
        self.proposal_network = ProposalNetwork(
            gaussian_init_min=gaussian_init_min,
            gaussian_init_max=gaussian_init_max,
            learning_rate_start=proposal_learning_rate_start,
            learning_rate_stop=proposal_learning_rate_stop,
            learning_rate_decay_end=proposal_learning_rate_decay_end,
            gaussian_learning_rate_factor=gaussian_learning_rate_factor,
            weight_decay=proposal_weight_decay
        )

        # Nerf network estimates colors and densities for volume rendering
        self.radiance_network = RadianceNetwork(
            gaussian_init_min=gaussian_init_min,
            gaussian_init_max=gaussian_init_max,
            learning_rate_start=radiance_learning_rate_start,
            learning_rate_stop=radiance_learning_rate_stop,
            learning_rate_decay_end=radiance_learning_rate_decay_end,
            gaussian_learning_rate_factor=gaussian_learning_rate_factor,
            weight_decay=radiance_weight_decay
        )

        # Transmittance estimator
        self.transmittance_estimator = nerfacc.PropNetEstimator()

        self.camera_extrinsics = CameraExtrinsics(
            n_train_images=n_train_images,
            learning_rate_start=camera_learning_rate_start,
            learning_rate_stop=camera_learning_rate_stop,
            learning_rate_decay_end=camera_learning_rate_decay_end,
        )

        self.param_groups = [
            param_group 
            for model in [self.proposal_network, self.radiance_network, self.camera_extrinsics] 
            for param_group in model.param_groups
        ]


    def _get_positions(
        self, 
        ray_origs: th.Tensor, 
        ray_dirs: th.Tensor,
        t_starts: th.Tensor,
        t_ends: th.Tensor
    ) -> th.Tensor:
        """Calculate the positions of the rays at the given t values.
        
        Args:
            ray_origs (th.Tensor): Tensor of shape (n_rays, 3) - the ray origins.
            ray_dirs (th.Tensor): Tensor of shape (n_rays, 3) - the ray directions.
            t_starts (th.Tensor): Tensor of shape (n_rays, n_samples) - the start values for each ray.
            t_ends (th.Tensor): Tensor of shape (n_rays, n_samples) - the end values for each ray.
            
        Returns:
            th.Tensor: Tensor of shape (n_rays, n_samples, 3) - the positions of the rays at the given t values.
        """
        # Calculate ray positions (n_rays, n_samples, 3)
        #   Origins and directions are reshaped to (n_rays, 1, 3) for broadcasting.
        #   t_starts and t_ends are reshaped to (n_rays, n_samples, 1) for broadcasting.
        #   The singleton dimensions in the tensors will then broadcast to (n_rays, n_samples, 3).
        return ray_origs[:, None] + ray_dirs[:, None] * ((t_starts + t_ends))[..., None] / 2


    def _create_proposal_forward(
        self, 
        ray_origs: th.Tensor, 
        ray_dirs: th.Tensor
    ) -> Callable[[th.Tensor, th.Tensor], th.Tensor]:
        """Create a closure that captures the current rays and returns a function that samples the density at the given t values.
        
        Args:
            ray_origs (th.Tensor): Tensor of shape (n_rays, 3) - the ray origins.
            ray_dirs (th.Tensor): Tensor of shape (n_rays, 3) - the ray directions.
            
        Returns:
            Callable[[th.Tensor, th.Tensor], th.Tensor]: Function that samples the density at the given t values.
        """
        # Define forward function that captures the current rays        
        def forward(t_starts: th.Tensor, t_ends: th.Tensor) -> th.Tensor:
            """Sample each ray at the given t values and return the sampled density.
            
            Args:
                t_starts (th.Tensor): Tensor of shape (n_rays, n_samples) - the start values for each ray.
                t_ends (th.Tensor): Tensor of shape (n_rays, n_samples) - the end values for each ray.
                
            Returns:
                th.Tensor: Tensor of shape (n_rays, n_samples) - the sampled density.
            """
            # Calculate positions to sample (n_rays, n_samples, 3)
            positions = self._get_positions(ray_origs, ray_dirs, t_starts, t_ends)
            
            # Calculate and return densities (n_rays, n_samples)
            return self.proposal_network(positions.view(-1, 3)).view(t_starts.shape)

        # Return closure
        return forward


    def _create_radiance_forward(
        self, 
        ray_origs: th.Tensor, 
        ray_dirs: th.Tensor
    ) -> Callable[[th.Tensor, th.Tensor, Optional[th.Tensor]], tuple[th.Tensor, th.Tensor]]:
        """Create a closure that captures the current rays and returns a function that samples the radiance at the given t values.
        
        Args:
            ray_origs (th.Tensor): Tensor of shape (n_rays, 3) - the ray origins.
            ray_dirs (th.Tensor): Tensor of shape (n_rays, 3) - the ray directions.
            
        Returns:
            Callable[[th.Tensor, th.Tensor, Optional[th.Tensor]], tuple[th.Tensor, th.Tensor]]: Function that samples the radiance at the given t values.
        """
        
        # Define forward function that captures the current rays
        def forward(
            t_starts: th.Tensor, 
            t_ends: th.Tensor, 
            ray_indices: Optional[th.Tensor] = None
        ) -> tuple[th.Tensor, th.Tensor]:
            """Sample each ray at the given t values and return the sampled density.
            
            Args:
                t_starts (th.Tensor): Tensor of shape (n_rays, n_samples) - the start values for each ray.
                t_ends (th.Tensor): Tensor of shape (n_rays, n_samples) - the end values for each ray.
                ray_indices (th.Tensor): Not used. Tensor of shape (n_rays, n_samples) - the ray indices for each sample.
                
            Returns:
                th.Tensor, th.Tensor: Tensors of shape (n_rays, n_samples, x), where x=3 for rgb, and x=1 for density.
            """
            # Calculate positions to sample (n_rays, n_samples, 3)
            positions = self._get_positions(ray_origs, ray_dirs, t_starts, t_ends)

            # Calculate rgb and densities (n_rays, n_samples, x)
            rgb, density = self.radiance_network(
                positions.view(-1, 3), 
                ray_dirs.repeat_interleave(positions.shape[1], dim=0)
            )
            
            # Return for volume rendering (n_rays, 3), (n_rays, n_samples)
            return rgb.view(*t_starts.shape, 3), density.view(t_starts.shape)

        # Return closure
        return forward


    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor, pixel_width: th.Tensor=None) -> tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """
        Forward pass of the model.
        Given the ray origins and directions, compute the rgb values for the given rays.

        Args:
            ray_origs: Tensor of shape (n_rays, 3) - focal point position in world, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (n_rays, 3) - direction vectors (unit vectors) of the viewing directions
            pixel_width: Tensor of shape (n_rays, 1) - NOT USED. the width of the pixel in world coordinates

        Returns:
            rgb: Tensor of shape (n_rays, 3) - the rgb values of the given rays
            opacity: Tensor of shape (n_rays, 1) - the opacity values of the given rays
            depth: Tensor of shape (n_rays, 1) - the depth values of the given rays
            extras: Dict - extra intermediate calculation data, e.g. transmittance
        """
        # Estimate positions to sample
        t_starts, t_ends = self.transmittance_estimator.sampling(
            prop_sigma_fns=[self._create_proposal_forward(ray_origs, ray_dirs)], 
            prop_samples=[self.proposal_samples_per_ray],
            num_samples=self.radiance_samples_per_ray,
            n_rays=ray_origs.shape[0],
            near_plane=self.near_sphere_normalized,
            far_plane=self.far_sphere_normalized,
            sampling_type="lindisp",
            stratified=self.training,
            requires_grad=th.is_grad_enabled()
        )

        # Sample colors and densities
        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts=t_starts,
            t_ends=t_ends,
            ray_indices=None,
            n_rays=None,
            rgb_sigma_fn=self._create_radiance_forward(ray_origs, ray_dirs),
            render_bkgd=None
        )


        # Return colors for the given pixel coordinates (n_rays, 3),
        #  together with the opacity (n_rays, 1) and depth (n_rays, 1),
        #  and the extras Dict
        return rgb, opacity, depth, extras




    def _forward_loss(self, batch: InnerModelBatchInput) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
        """Forward pass that also calculates losses.
        
        Args:
            batch (InnerModelBatchInput): Batch of data

        Returns:
            tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]: Color predictions and ordered losses for each optimizer
        """
        # Decontsruct batch
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
        ray_colors_pred, ray_opacity, ray_depth, extras = self(ray_origs_pred, ray_dirs_pred)

        # Calculate losses for training
        proposal_loss = self.transmittance_estimator.compute_loss(extras["trans"].detach())
        # NOTE Not using blurred pixel colors
        radiance_loss = F.mse_loss(ray_colors_pred, ray_colors_raw[:, -1, ...]) 

        # Return color prediction and losses
        return (
            ray_colors_pred,
            (proposal_loss, radiance_loss)
        )



    def _get_logging_losses(
        self, 
        stage: Literal["train", "val", "test"], 
        proposal_loss: th.Tensor, 
        radiance_loss: th.Tensor,
        *args, 
        **kwargs
    ) -> dict[str, th.Tensor]:
        # Calculate PSNR
        # NOTE: Cannot calculate SSIM because it relies on image patches
        # NOTE: Cannot calculate LPIPS because it relies on image patches
        psnr = self.compute_psnr(radiance_loss)

        # Return losses to be logged
        return {
            f"{stage}_proposal_loss": proposal_loss.cpu().detach(),
            f"{stage}_radiance_loss": radiance_loss.cpu().detach(),
            f"{stage}_psnr": psnr
        }
        


    def training_step(self, batch: InnerModelBatchInput, batch_idx: int):
        """Perform a single training forward pass and optimization step.
        
        Args:
            batch (InnerModelBatchInput): Batch of data
            batch_idx (int): Index of the batch
        
        Returns:
            th.Tensor: Loss
        """

        batch = self.training_transform(batch)
        

        # Forward pass
        _, (proposal_loss, radiance_loss) = self._forward_loss(batch)


        # Get logging losses
        logs = self._get_logging_losses(
            "train",
            proposal_loss,
            radiance_loss
        )
        
        # Rate limit pose error logging
        if batch_idx % 100 == 0:
            pose_error = self.compute_pose_error() 
            logs["pose_error"] = pose_error.cpu().detach()

        # Log metrics
        self.log_dict(logs)


        # Return loss
        return radiance_loss + proposal_loss


    def validation_step(self, batch: InnerModelBatchInput, batch_idx: int):
        """Perform a single validation forward pass.
        
        Args:
            batch (InnerModelBatchInput): Batch of data
            batch_idx (int): Index of the batch
        
        Returns:
            th.Tensor: Loss
        """
        
        batch = self.validation_transform(batch)
        
        # Forward pass
        _, (proposal_loss, radiance_loss) = self._forward_loss(batch)


        # Get logging losses
        logs = self._get_logging_losses(
            "val",
            proposal_loss,
            radiance_loss
        )

        # Rate limit pose error logging
        if batch_idx == 0:
            pose_error = self.compute_pose_error() 
            logs["pose_error"] = pose_error

        # Log metrics
        self.log_dict(logs)


        return radiance_loss + proposal_loss



