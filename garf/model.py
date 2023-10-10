from typing import Callable, Optional, Dict

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import nerfacc

from data_module import DatasetOutput
from model_radiance import RadianceNetwork
from model_proposal import ProposalNetwork


class Garf(pl.LightningModule):
    def __init__(
        self, 
        near_plane: float,
        far_plane: float,
        proposal_samples_per_ray: int,
        radiance_samples_per_ray: int,
        gaussian_init_min: float = 0.0,
        gaussian_init_max: float = 1.0,
        proposal_learning_rate: float = 1e-4,
        proposal_learning_rate_decay: float = 0.5,
        proposal_weight_decay: float = 0.0,
        radiance_learning_rate: float = 1e-4,
        radiance_learning_rate_decay: float = 0.5,
        radiance_weight_decay: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # The near and far sphere distances 
        self.near_sphere_normalized = near_plane
        self.far_sphere_normalized = far_plane

        # Samples per ray for proposal network for the density estimation
        self.proposal_samples_per_ray = proposal_samples_per_ray
        # Samples per ray for radiance network for the volumetric rendering
        self.radiance_samples_per_ray = radiance_samples_per_ray
        
        # Gaussian variance for the activation function
        self.gaussian_init_min = gaussian_init_min
        self.gaussian_init_max = gaussian_init_max

        # Hyper parameters for the network training
        self.proposal_learning_rate = proposal_learning_rate
        self.proposal_learning_rate_decay = proposal_learning_rate_decay
        self.proposal_weight_decay = proposal_weight_decay
        self.radiance_learning_rate = radiance_learning_rate
        self.radiance_learning_rate_decay = radiance_learning_rate_decay
        self.radiance_weight_decay = radiance_weight_decay

        # Proposal network estimates sampling density
        self.proposal_network = ProposalNetwork(
            gaussian_init_min=gaussian_init_min,
            gaussian_init_max=gaussian_init_max,
        )

        # Nerf network estimates colors and densities for volume rendering
        self.radiance_network = RadianceNetwork(
            gaussian_init_min=gaussian_init_min,
            gaussian_init_max=gaussian_init_max,
        )
        
        
        # Transmittance estimator
        self.transmittance_estimator = nerfacc.PropNetEstimator()
        
        
        # Enable manual optimization
        self.automatic_optimization = False


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


    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, Dict]:
        """
        Forward pass of the model.
        Given the ray origins and directions, compute the rgb values for the given rays.

        Parameters:
        -----------
            ray_origs: Tensor of shape (n_rays, 3) - focal point position in world, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (n_rays, 3) - direction vectors (unit vectors) of the viewing directions

        Returns:
        --------
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        """
        
        
        
        
        
        # TODO: Finish documentation
        
        
        
        
        
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


        # Return colors for the given pixel coordinates (n_rays, 3)
        return rgb, opacity, depth, extras




    def _forward_loss(self, batch: DatasetOutput, batch_idx: int, stage: str):
        ray_origs, ray_dirs, ray_colors = batch
        
        ray_colors_pred, ray_opacity, ray_depth, extras = self(ray_origs, ray_dirs)



        # TODO: FIX ME
        proposal_loss = self.transmittance_estimator.compute_loss(extras["trans"])
        radiance_loss = nn.functional.mse_loss(ray_colors_pred, ray_colors)






        loss = proposal_loss + radiance_loss
        # TODO: Log all the losses, including new psnr and stuff
        
        
        
        
        
        
        
        
        self.log(f"{stage}_loss", loss)

        # NOTE: Assuming losses are ordered according to associated optimizer
        return proposal_loss, radiance_loss

    def training_step(self, batch: DatasetOutput, batch_idx: int):
        # Get optimizers for manual optimization
        # NOTE: Technically getting a list of LightningOptimizer wrappers,
        #  but they have not set up typing correctly. Expect more type ignores.
        optimizers: list[th.optim.Optimizer] = self.optimizers() # type: ignore

        # Forward pass
        # NOTE: Assuming losses are ordered according to associated optimizers
        losses = self._forward_loss(batch, batch_idx, "train")

        # Backward pass and step through each optimizer
        for loss, optimizer in zip(losses, optimizers):
            optimizer.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            optimizer.step()

        # Return summed loss
        return sum(losses)
        

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        return self._forward_loss(batch, batch_idx, "val")


    def configure_optimizers(self):
        # Set up proposal optimizers
        proposal_optimizer = th.optim.Adam(
            self.proposal_network.parameters(), 
            lr=self.proposal_learning_rate, 
            weight_decay=self.proposal_weight_decay,
        )
        proposal_scheduler = th.optim.lr_scheduler.ExponentialLR(
            proposal_optimizer, 
            gamma=self.proposal_learning_rate_decay
        )
        
        # Set up radiance optimizers
        radiance_optimizer = th.optim.Adam(
            self.radiance_network.parameters(), 
            lr=self.radiance_learning_rate, 
            weight_decay=self.radiance_weight_decay,
        )
        radiance_scheduler = th.optim.lr_scheduler.ExponentialLR(
            radiance_optimizer, 
            gamma=self.radiance_learning_rate_decay
        )


        # Set optimizers and schedulers
        # NOTE: Assuming losses are ordered according to associated optimizer
        return [proposal_optimizer, radiance_optimizer], [proposal_scheduler, radiance_scheduler]


