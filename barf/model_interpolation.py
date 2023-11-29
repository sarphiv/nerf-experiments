from typing import Literal
from itertools import chain

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from model_interpolation_architecture import NerfModel, PositionalEncoding

# Type alias for inner model batch input
#  (origin, direction, pixel_color, pixel_relative_blur)
InnerModelBatchInput = tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]

# MAGIC_NUMBER = 7#/24
from magic import MAGIC_NUMBER

class NerfInterpolation(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray: int,
        n_hidden: int,
        hidden_dim: int,
        proposal: tuple[bool, int],
        position_encoder: PositionalEncoding,
        direction_encoder: PositionalEncoding,
        delayed_direction: bool, 
        delayed_density: bool, 
        n_segments: int,
        learning_rate_start: float = 1e-4,
        learning_rate_stop: float = 1e-4,
        learning_rate_stop_step: int = -1,
        weight_decay: float = 0.0
    ):  


        super().__init__()
        # self.save_hyperparameters()

        # Hyper parameters for the optimizer 
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.learning_rate_stop_step = learning_rate_stop_step


        self.weight_decay = weight_decay

        # The near and far sphere distances 
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        # the encoders
        self.position_encoder = position_encoder
        self.direction_encoder = direction_encoder

        # Coarse model -> proposal network, but when there is no proposal network it is the actual prediction
        self.model_coarse = NerfModel(
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            delayed_direction=delayed_direction,
            delayed_density=delayed_density,
            n_segments=n_segments,
            position_encoder=position_encoder,
            direction_encoder=direction_encoder)
        

        # If there is a proposal network separate the samples into coarse and fine sampling
        self.proposal = proposal[0]
        if self.proposal:
            self.samples_per_ray_coarse = proposal[1]
            self.samples_per_ray_fine = samples_per_ray - proposal[1]

            # Fine model -> actual prediction 
            self.model_fine = NerfModel(
                n_hidden=n_hidden,
                hidden_dim=hidden_dim,
                delayed_direction=delayed_direction,
                delayed_density=delayed_density,
                n_segments=n_segments,
                position_encoder=position_encoder,
                direction_encoder=direction_encoder)
            
            # Define the prediction network 
            self.model_prediction = self.model_fine

        else: 
            self.samples_per_ray_coarse = samples_per_ray
            self.samples_per_ray_fine = 0

            # Define the prediction network 
            self.model_prediction = self.model_coarse

            # List of models 


    def _get_intervals(self, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        From t-values get the t corresponding to the start and end of the bins.
        
        Parameters:
        -----------
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.
            
        Returns:
        --------
            t_start: Tensor of shape (batch_size, samples_per_ray) - the t value for the beginning of the bin
            t_end: Tensor of shape (batch_size, samples_per_ray) - the t value for the end of the bin
        """
        t_start = t 
        t_end = th.zeros_like(t, device=self.device)
        t_end[:,:-1] = t[:, 1:].clone()
        t_end[:,-1] = self.far_sphere_normalized

        return t_start, t_end


    def _sample_t_coarse(self, batch_size: int) -> tuple[th.Tensor, th.Tensor]:
        """
        Sample t for coarse sampling. Divides interval into equally sized bins, and samples a random point in each bin.

        Parameters:
        -----------
            batch_size: int - the amount of rays in a batch

        Returns:
        -----------
            t_start: Tensor of shape (batch_size, samples_per_ray) - start of the t-bins
            t_end: Tensor of shape (batch_size, samples_per_ray) - end of the t-bins
        """
        # n = samples_per_ray_coarse 
        # Calculate the interval size (divide far - near into n bins)
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / self.samples_per_ray_coarse
        
        # Sample n points with equal distance, starting at the near sphere and ending at one interval size before the far sphere
        t_coarse = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray_coarse, 
            device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)
        
        # Perturb sample positions to be anywhere in each interval
        t_coarse += th.rand((batch_size, 1), device=self.device) * interval_size

        return self._get_intervals(t_coarse)
    

    def _sample_t_fine(self, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor) ->  tuple[th.Tensor, th.Tensor]:
        """
        Using the coarsely sampled t values to decide where to sample more densely. 
        The weights express how large a percentage of the light is blocked by that specific segment, hence that percentage of the new samples should be in that segment.

        Parameters:
        -----------
            t_coarse:           Tensor of shape (batch_size, samples_per_ray_proposal) - the t-values for the beginning of each bin
            weights:            Tensor of shape (batch_size, samples_per_ray_proposal) (these are assumed to almost sum to 1)
            distances_coarse:   Tensor of shape (batch_size, samples_per_ray_proposal)
        
        Returns:
        --------
            t_start:            Tensor of shape (batch_size, samples_per_ray) - start of the t-bins
            t_end:              Tensor of shape (batch_size, samples_per_ray) - end of the t-bins
        
        """
        # Initialization 
        batch_size, n_bins = t_coarse.shape
        device = t_coarse.device
        total_samples = self.samples_per_ray_fine + n_bins 

        # Each segment needs weight*samples_per_ray_fine new samples plus 1 because of the coarse sample 
        weights_rescaled = weights/weights.sum(dim=1, keepdim=True)
        fine_samples_raw = weights_rescaled*(total_samples - n_bins) # Get the number of extra fine samples for each bin 
        fine_samples = th.floor(fine_samples_raw)                    # Floor to get an under estimate 
        fine_samples_errors = fine_samples_raw - fine_samples        # Get how much each bin needs an extra sample 
        excess = total_samples - n_bins - fine_samples.sum(dim=1, keepdim=True)# Find how many more samples needs to be in added in the different bins
        
        # Place the excess samples in the correct bins 
        error_rank = fine_samples_errors.argsort(dim=1).argsort(dim=1)  # Get the rank of the errors
        add_mask = (error_rank >= (n_bins - excess)) # We have n_bins and the excess' highest ranks needs 1 more. Those are taken by comparing ranks to n_bins - excess
        fine_samples = fine_samples + add_mask + 1 # Add mask and the coarse samples to each bin 

        # We need to know how many samples have been before the k'th bin 
        fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1, device=device), fine_samples.cumsum(dim=1)))

        # Instanciate the t_fine tensor and arange is used to mask the correct t values for each batch
        arange = th.arange(total_samples, device=device).unsqueeze(0)
        t_fine = th.zeros(batch_size, total_samples, device=device)

        for i in range(n_bins):
            # Pick out the samples for each segment, everything within cumsum[i] and cumsum[i+1] is in segment i because the difference is the new samples
            # This mask is for each ray in the batch 
            mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
            # Set samples to be the coarsely sampled point 
            t_fine += t_coarse[:, i].unsqueeze(-1)*mask
            # And spread them out evenly in the segment by deviding the length of the segment with the amount of samples in that segment
            t_fine += (arange - fine_samples_cum_sum[:, i].unsqueeze(-1))*mask*distances_coarse[:, i].unsqueeze(-1)/fine_samples[:, i].unsqueeze(-1)

        # If against all odds there are any zeroes in t_fine, we replace them with the far sphere
        if (t_fine == 0).any():
            t_fine[t_fine == 0] = self.far_sphere_normalized
            print(f"There were zeroes in t_fine?{weights}")

        t_start, t_end = self._get_intervals(t_fine)
        
        return t_start, t_end


    def _compute_positions(self, origins: th.Tensor, directions: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Compute the positions and directions for given rays and t values.

        Parameters:
        -----------
            origins: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            directions: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.


        Returns:
        --------
            positions: Tensor of shape (batch_size, samples_per_ray, 3) - the positions for the given t values
            directions: Tensor of shape (batch_size, samples_per_ray, 3) - the directions for the given t values
        """
        # Calculates the position with the equation p = o + t*d 
        # The unsqueeze is because origins and directions are spacial vectors while t is the sample times
        positions = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)

        # Format the directions to match the found positions
        directions = directions.unsqueeze(1).repeat(1, positions.shape[1], 1)
        
        return positions, directions


    def _render_rays(self, densities: th.Tensor, colors: th.Tensor, distances: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Render the rays using the given densities and colors.
        I.e. this is just an implementation of equation 3.
        blocking_neg    = -sigma_j * delta_j
        alpha_j         = 1 - exp(blocking_neg)
        alpha_int_i     = T_i = exp(sum_{j=1}^{i-1} blocking_neg_j)
        output          = sum_{j=1}^N alpha_int_j * alpha_j * c_j (c_j = color_j)

        Parameters:
        -----------
            densities: Tensor of shape (batch_size, samples_per_ray) - the densities for the given samples (sigma_j in the paper)
            colors: Tensor of shape (batch_size, samples_per_ray, 3) - the colors for the given samples (c_j in the paper)
            distances: Tensor of shape (batch_size, samples_per_ray) - the distances between the samples (delta_j in the paper)
        
        
        Returns:
        --------
            rgb: Tensor of shape (batch_size, 3)                    - the rgb values for the given rays
            weights: Tensor of shape (batch_size, samples_per_ray)  - the weights used for the fine sampling

        """

        # Get the negative Optical Density (needs to be weighted up, because the scene is small)
        blocking_neg = (-densities * distances)*3*MAGIC_NUMBER
        # Get the absorped light over each ray segment 
        alpha = 1 - th.exp(blocking_neg)
        # Get the light that has made it through previous segments 
        alpha_int = th.hstack((
            th.ones((blocking_neg.shape[0], 1), device=self.device),
            th.exp(th.cumsum(blocking_neg[:, :-1], dim=1))
        ))

        # Weight that express how much each segment influences the final color 
        weights = alpha_int * alpha
        
        # Compute the final color by summing over the weighted colors (the unsqueeze(-1) is to mach dimensions)
        return th.sum(weights.unsqueeze(-1)*colors, dim=1), weights


    def _compute_color(self, model: NerfModel,
                            t_start: th.Tensor,
                            t_end: th.Tensor,
                            ray_origs: th.Tensor,
                            ray_dirs: th.Tensor,
                            batch_size: int,
                            samples_per_ray: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        """
        A helper function to compute the color for given model, t values, ray origins and ray directions.
        Is just a wrapper around _compute_positions, model_base.forward and _render_rays.

        Parameters:
        -----------
            model: The model to use for computing the color (either the coarse or fine model)
            t_start: Tensor of shape (batch_size, samples_per_ray) - the t value for the begining of the bin
            t_end: Tensor of shape (batch_size, samples_per_ray) - the t value for the end of the bin
            ray_origs: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions
            batch_size: int - the batch size
            samples_per_ray: int - the number of samples per ray
        
        Returns:
        --------
            rgb: Tensor of shape (batch_size, 3) - the rgb values for the given rays
            weights: Tensor of shape (batch_size, samples_per_ray) - the weights used for the fine sampling
            sample_dist: Tensor of shape (batch_size, samples_per_ray) - the distances between the samples (delta_j in the paper)
        
        """
        # Compute the positions for the given t values
        sample_pos, sample_dir  = self._compute_positions(ray_origs, ray_dirs, (t_start + t_end)/2)
        sample_dist = t_end - t_start

        # Ungroup samples by ray (for the model)
        sample_pos = sample_pos.view(batch_size * samples_per_ray, 3)
        sample_dir = sample_dir.view(batch_size * samples_per_ray, 3)
        
        # Evaluate density and color at sample positions
        sample_density, sample_color = model(sample_pos, sample_dir)

        assert not th.isnan(sample_density).any(), "Density is NaN"
        assert not th.isnan(sample_color).any(), "Color is NaN"
        
        # Group samples by ray
        sample_density = sample_density.view(batch_size, samples_per_ray)
        sample_color = sample_color.view(batch_size, samples_per_ray, 3)

        # Compute the rgb of the rays, and the weights for fine sampling
        rgb, weights = self._render_rays(sample_density, sample_color, sample_dist)
        
        return rgb, weights, sample_dist

        

    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the model.
        Given the ray origins and directions, compute the rgb values for the given rays.

        Parameters:
        -----------
            ray_origs: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions

        Returns:
        --------
            rgb_fine: Tensor of shape (batch_size, 3) - the rgb values for the given rays using fine model (the actual prediction)
            rgb_coarse: Tensor of shape (batch_size, 3) - the rgb values for the given rays using coarse model (only used for the loss - not the rendering)
        """

        # Amount of pixels to render
        batch_size = ray_origs.shape[0]
        
        
        ### Coarse sampling
        # sample t
        t_coarse_start, t_coarse_end = self._sample_t_coarse(batch_size)

        #compute rgb
        rgb_coarse, weights, sample_dist_coarse = self._compute_color(self.model_coarse,
                                                t_coarse_start, 
                                                t_coarse_end,
                                                ray_origs,
                                                ray_dirs,
                                                batch_size,
                                                self.samples_per_ray_coarse)
        
        if self.proposal:
            ### Fine sampling
            # Sample t
            t_fine_start, t_fine_end = self._sample_t_fine(t_coarse_start, weights, sample_dist_coarse)
            # compute rgb
            rgb_fine, _, _ = self._compute_color(self.model_fine,
                                                    t_fine_start, 
                                                    t_fine_end,
                                                    ray_origs,
                                                    ray_dirs,
                                                    batch_size,
                                                    self.samples_per_ray_coarse + self.samples_per_ray_fine)
        else: 
            rgb_fine = rgb_coarse
            rgb_coarse = th.zeros_like(rgb_coarse)

        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb_fine, rgb_coarse

    def validation_image(self, ray_origs: th.Tensor, ray_dirs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the model for a validation step. In vanilla nerf this is equivalent to the usual forward pass. 
        """
        return self.forward(ray_origs, ray_dirs)

    ############ pytorch lightning functions ############
    
    
    # def _forward_loss(self, batch: InnerModelBatchInput) -> tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]:
    #     """Forward pass that also calculates losses.
        
    #     Args:
    #         batch (InnerModelBatchInput): Batch of data

    #     Returns:
    #         tuple[th.Tensor, tuple[th.Tensor, th.Tensor]]: Color predictions and ordered losses for each optimizer
    #     """
    #     # Decontsruct batch
    #     ray_origs, ray_dirs, ray_colors, ray_scales = batch
        
    #     # Forward pass
    #     ray_colors_pred_fine, ray_colors_pred_coarse = self(ray_origs, ray_dirs)

    #     # Calculate losses for training
    #     proposal_loss = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors)
    #     radiance_loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors)

    #     # Return color prediction and losses
    #     return (
    #         ray_colors_pred_fine,
    #         (proposal_loss, radiance_loss)
    #     )
    
    
    # def training_step(self, batch: InnerModelBatchInput, batch_idx: int):
    #     """Perform a single training forward pass and optimization step.
        
    #     Args:
    #         batch (InnerModelBatchInput): Batch of data
    #         batch_idx (int): Index of the batch
        
    #     Returns:
    #         th.Tensor: Loss
    #     """
    #     # Set alpha value in the nerf models
    #     for model in self.models: 
    #         # TODO instead of using epoch fraction use global step count  
    #         # The fourier_levels_per_epoch is a hyperparameter 
    #         model.alpha = model.fourier_levels_start + (self.trainer.current_epoch + batch_idx/self.trainer.num_training_batches)*model.fourier_levels_per_epoch

    #     # Forward pass
    #     _, (proposal_loss, radiance_loss) = self._forward_loss(batch)

    #     # Backward pass and step through each optimizer
    #     self._proposal_optimizer_step(proposal_loss + radiance_loss)
    #     self._radiance_optimizer_step(proposal_loss + radiance_loss)

    #     # Step learning rate schedulers
    #     self._proposal_scheduler_step(batch_idx)
    #     self._radiance_scheduler_step(batch_idx)

    #     # Log metrics
    #     self.log_dict(self._get_logging_losses(
    #         "train",
    #         proposal_loss,
    #         radiance_loss,
    #     ))


    #     # Return loss
    #     return radiance_loss


    # def validation_step(self, batch: InnerModelBatchInput, batch_idx: int):
    #     """Perform a single validation forward pass.
        
    #     Args:
    #         batch (InnerModelBatchInput): Batch of data
    #         batch_idx (int): Index of the batch
        
    #     Returns:
    #         th.Tensor: Loss
    #     """
    #     # Forward pass
    #     _, (proposal_loss, radiance_loss) = self._forward_loss(batch)

    #     # Log metrics
    #     self.log_dict(self._get_logging_losses(
    #         "val",
    #         proposal_loss,
    #         radiance_loss,
    #     ))
        
    #     return radiance_loss


    # def configure_optimizers(self):
    #     self._optimizer = th.optim.Adam(
    #         chain(*(model.parameters() for model in self.models)),
    #         lr=self.learning_rate, 
    #         weight_decay=self.weight_decay,
    #     )
    #     self._scheduler = th.optim.lr_scheduler.ExponentialLR(
    #         self._optimizer, 
    #         gamma=self.learning_rate_decay
    #     )

    #     return (
    #         [self._optimizer],
    #         [self._scheduler],
    #     )