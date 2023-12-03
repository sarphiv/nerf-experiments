from typing import Literal, Callable, Optional
from itertools import chain
import warnings
import math
import os

import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from model_interpolation_architecture import NerfModel, PositionalEncoding


# Type alias for inner model batch input
#  (origin, direction, pixel_color, pixel_relative_blur)
InnerModelBatchInput = tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]

uniform_sampling_strategies = Literal["stratified_uniform", "equidistant"]
integration_strategies = Literal["left", "middle"]

# MAGIC_NUMBER = 7#/24
from magic import MAGIC_NUMBER


class DummyCamEx(nn.Module):
    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        return o, d, None, None

    
class SchedulerLeNice(th.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer: th.optim.Optimizer, start_LR: list[float], stop_LR: list[float] | None = None, number_of_steps: list[float] | None = None, verbose=False) -> None:
        # Store extra parameters 
        self.start_LR = start_LR
        self.stop_LR = stop_LR
        self.number_of_steps = number_of_steps

        # Calculate decay factors
        # Solve: start : s, end : e, decay : d, number of epochs : n 
        # s * d^n = e <=> d = (e/s)^(1/n)
        self.decay_factors = []
        self.log_decay_factors = []
        for i, _ in enumerate(optimizer.param_groups):
            if self.number_of_steps is None or self.number_of_steps[i] in [0, None] or self.start_LR[i] == 0:
                decay_factor = 1.
                log_decay_factor = 0.
            else:
                decay_factor = (self.stop_LR[i] / self.start_LR[i]) ** (1 / self.number_of_steps[i])
                log_decay_factor = 1/self.number_of_steps[i] * (math.log(self.stop_LR[i]) - math.log(self.start_LR[i]))

            self.decay_factors.append(decay_factor)
            self.log_decay_factors.append(log_decay_factor)

        super().__init__(optimizer,verbose=verbose)
        
    def get_lr(self):
        # Original function 
        return self._get_closed_form_lr()
        # if not self._get_lr_called_within_step:
        #     warnings.warn("To get the last learning rate computed by the scheduler, "
        #                   "please use `get_last_lr()`.", UserWarning)

        # # Update lr for each individually 
        # return [(group['lr'] * self.decay_factors[i] if self._step_count < self.number_of_steps[i] else group["lr"]) for i, group in enumerate(self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        # return [base_lr * self.decay_factors[i] ** min(self._step_count, self.number_of_steps[i]) for i, base_lr in enumerate(self.start_LR)]
        return [base_lr * math.exp(self.log_decay_factors[i] * min(self._step_count, self.number_of_steps[i])) for i, base_lr in enumerate(self.start_LR)]



class NerfInterpolation(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        model_radiance: NerfModel,
        samples_per_ray_radiance: int,
        uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
        uniform_sampling_offset_size: float = 0.,
        integration_strategy: integration_strategies = "middle",
        model_proposal: NerfModel|None = None,
        samples_per_ray_proposal: int = 0,

    ):  
        
        pl.LightningModule.__init__(self)

        self.save_hyperparameters(
            ignore=["model_radiance", "model_proposal",]
                                )

        # The near and far sphere distances 
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        self.samples_per_ray_radiance = samples_per_ray_radiance
        self.samples_per_ray_proposal = samples_per_ray_proposal

        self.uniform_sampling_strategy = uniform_sampling_strategy
        self.uniform_sampling_offset_size = uniform_sampling_offset_size
        self.integration_strategy = integration_strategy

        self.model_radiance = model_radiance
        self.model_proposal = model_proposal
        self.camera_extrinsics = DummyCamEx()

        self.proposal = samples_per_ray_proposal > 0

        self.param_groups = [param_group for model in ([model_radiance, model_proposal] if self.proposal else [model_radiance])
                             for param_group in model.param_groups]



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


    def _sample_t_stratified_uniform(self,
                                     batch_size: int,
                                     n_samples: int,
                                     strategy: uniform_sampling_strategies,
                                     offset_size: float,
                                     ) -> tuple[th.Tensor, th.Tensor]:
        """
        Sample t for coarse sampling. Divides interval into equally sized bins, and samples a random point in each bin.

        Parameters:
        -----------
            batch_size: int - the amount of rays in a batch
            n_samples: int - samples per ray
            strategy: str - what sampling strategy to use
            offset_size: float in [0,1] - offset all samples by the same uniformly random number in the interval [0, interval_size*offset_size]
                                          where interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / n_samples

        Returns:
        -----------
            t_start: Tensor of shape (batch_size, samples_per_ray) - start of the t-bins
            t_end: Tensor of shape (batch_size, samples_per_ray) - end of the t-bins
        """
        # n = samples_per_ray_coarse 
        # Calculate the interval size (divide far - near into n bins)
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / n_samples
        
        # Sample n points with equal distance, starting at the near sphere and ending at one interval size before the far sphere
        t_coarse = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            n_samples, 
            device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)

        if strategy == "stratified_uniform":
            # Perturb sample positions to be anywhere in each interval
            t_coarse += th.rand((batch_size, n_samples), device=self.device) * interval_size
        elif strategy == "equidistant":
            pass
        else:
            raise ValueError(f"sampling_strategy must be one of {uniform_sampling_strategies.__args__}, was '{strategy}'")

        if offset_size != 0:
            t_coarse += th.rand((batch_size, 1), device=self.device) * interval_size * offset_size

        return self._get_intervals(t_coarse)
    
    # def _sample_t_pdf_weighted(self, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor, n_samples: int) ->  tuple[th.Tensor, th.Tensor]:
    #     batch_size, n_bins = t_coarse.shape
    #     device = t_coarse.device
    #     pdf = weights/weights.sum(dim=1, keepdim=True)
    #     cdf = pdf.cumsum(dim=1)
    #     u = th.linspace(0,1,n_samples - n_bins, device=device).unsqueeze(0)
    #     idx = th.searchsorted(cdf, u.repeat(batch_size, 1))




    def _sample_t_pdf_weighted(self, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor, n_samples: int) ->  tuple[th.Tensor, th.Tensor]:
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

        # Each segment needs weight*samples_per_ray_fine new samples plus 1 because of the coarse sample 
        weights_rescaled = weights/weights.sum(dim=1, keepdim=True)
        fine_samples_raw = weights_rescaled*(n_samples - n_bins)
        fine_samples = th.floor(fine_samples_raw)
        fine_samples_errors = fine_samples_raw - fine_samples

        # fine_samples[th.arange(batch_size), th.argmax(fine_samples, dim=1)] += n_samples - fine_samples.sum(dim=1) - n_bins
        # fine_samples += 1

        fine_samples_sum = fine_samples.sum(dim=1, keepdim=True)
        excess = n_samples - n_bins - fine_samples_sum
        error_rank = fine_samples_errors.argsort(dim=1).argsort(dim=1)
        add_mask = (error_rank >= (n_bins - excess))
        fine_samples = fine_samples + add_mask + 1


        fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1, device=device), fine_samples.cumsum(dim=1)))


        sample_pdf_succeeded = False
        for i in range(5):
            if th.any(fine_samples < 0) or th.any(fine_samples_cum_sum[:, -1] != n_samples):
                if i == 0:
                    save_dict = {
                        "t_coarse": t_coarse, 
                        "weights": weights, 
                        "distances_coarse": distances_coarse, 
                        "n_samples": n_samples
                        }
                    dirpath = os.path.join(self.trainer.logger.experiment.dir, f"pdf_sampling_snapshot_{self.trainer.global_step}")
                    th.save(save_dict, dirpath)
                    print(f"pdf sampling didn't work as expected - trying again. See {dirpath}")
                
                fine_samples_sum = fine_samples.sum(dim=1, keepdim=True)
                excess = n_samples - n_bins - fine_samples_sum
                error_rank = fine_samples_errors.argsort(dim=1).argsort(dim=1)
                add_mask = (error_rank >= (n_bins - excess))
                fine_samples = fine_samples + add_mask + 1
            else:
                sample_pdf_succeeded = True
                break

        if sample_pdf_succeeded:
            
            # Instanciate the t_fine tensor and arange is used to mask the correct t values for each batch
            arange = th.arange(n_samples, device=device).unsqueeze(0)
            t_fine = th.zeros(batch_size, n_samples, device=device)

            for i in range(n_bins):
                # Pick out the samples for each segment, everything within cumsum[i] and cumsum[i+1] is in segment i because the difference is the new samples
                # This mask is for each ray in the batch 
                mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
                # Set samples to be the coarsely sampled point 
                t_fine += t_coarse[:, i].unsqueeze(-1)*mask
                # And spread them out evenly in the segment by deviding the length of the segment with the amount of samples in that segment
                t_fine += (arange - fine_samples_cum_sum[:, i].unsqueeze(-1))*mask*distances_coarse[:, i].unsqueeze(-1)/fine_samples[:, i].unsqueeze(-1)

            t_start, t_end = self._get_intervals(t_fine)
        
        else:
            print("pdf_sampling failed after 5 tries - using _sample_t_stratified_uniform instead")
            t_start, t_end = self._sample_t_stratified_uniform(batch_size, n_samples, "equidistant", -1)

        return t_start, t_end

    def _get_t_query(self, t_start: th.Tensor, t_end: th.Tensor, strategy: integration_strategies) -> th.Tensor:
        if strategy == "left":
            t_query = t_start
        elif strategy == "middle":
            t_query = (t_start + t_end)/2
        else:
            raise ValueError(f"strategy must be one of {integration_strategies.__args__}, was '{strategy}'")
        return t_query

    def _compute_positions(self, origins: th.Tensor, directions: th.Tensor, t_start: th.Tensor, t_end: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
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
        t = self._get_t_query(t_start, t_end, self.integration_strategy)
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
                            pixel_width: th.Tensor,
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
        sample_pos, sample_dir = self._compute_positions(ray_origs, ray_dirs, t_start, t_end)
        sample_pixel_width = pixel_width.repeat(1, samples_per_ray)
        sample_dist = t_end - t_start


        # Ungroup samples by ray (for the model)
        sample_pixel_width = sample_pixel_width.view(batch_size * samples_per_ray, 1)
        sample_pos = sample_pos.view(batch_size * samples_per_ray, 3)
        sample_dir = sample_dir.view(batch_size * samples_per_ray, 3)
        sample_t_start = t_start.view(batch_size * samples_per_ray, 1)
        sample_t_end = t_end.view(batch_size * samples_per_ray, 1)
        

        # Evaluate density and color at sample positions
        sample_density, sample_color = model.forward(sample_pos, sample_dir, sample_pixel_width, sample_t_start, sample_t_end)

        if th.isnan(sample_density).any(): print("Density is NaN")
        if th.isnan(sample_color).any(): print("Color is NaN")
        
        # Group samples by ray
        sample_density = sample_density.view(batch_size, samples_per_ray)
        sample_color = sample_color.view(batch_size, samples_per_ray, 3)

        # Compute the rgb of the rays, and the weights for fine sampling
        rgb, weights = self._render_rays(sample_density, sample_color, sample_dist)
        

        return rgb, weights, sample_dist


    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor, pixel_width: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forward pass of the model.
        Given the ray origins and directions, compute the rgb values for the given rays.

        Parameters:
        -----------
            ray_origs: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions
            pixel_width: Tensor of shape (batch_size, 3) - pixel widths at distance 1 from the camera

        Returns:
        --------
            rgb_fine: Tensor of shape (batch_size, 3) - the rgb values for the given rays using fine model (the actual prediction)
            rgb_coarse: Tensor of shape (batch_size, 3) - the rgb values for the given rays using coarse model (only used for the loss - not the rendering)
        """

        # Amount of pixels to render
        batch_size = ray_origs.shape[0]
        
        
        if self.proposal:
            ### Coarse sampling
            # sample t
            t_coarse_start, t_coarse_end = self._sample_t_stratified_uniform(batch_size,
                                                                             self.samples_per_ray_proposal,
                                                                             self.uniform_sampling_strategy,
                                                                             self.uniform_sampling_offset_size)

            #compute rgb
            rgb_coarse, weights, sample_dist_coarse = self._compute_color(self.model_proposal,
                                                    t_coarse_start, 
                                                    t_coarse_end,
                                                    ray_origs,
                                                    ray_dirs,
                                                    pixel_width,
                                                    batch_size,
                                                    self.samples_per_ray_proposal)
        
            ### Fine sampling
            # Sample t
            t_fine_start, t_fine_end = self._sample_t_pdf_weighted(t_coarse_start, weights, sample_dist_coarse, self.samples_per_ray_radiance)
            # compute rgb
            rgb_fine, _, _ = self._compute_color(self.model_radiance,
                                                    t_fine_start, 
                                                    t_fine_end,
                                                    ray_origs,
                                                    ray_dirs,
                                                    pixel_width,
                                                    batch_size,
                                                    self.samples_per_ray_radiance)
        else: 
            t_fine_start, t_fine_end = self._sample_t_stratified_uniform(batch_size,
                                                                         self.samples_per_ray_radiance,
                                                                         self.uniform_sampling_strategy,
                                                                         self.uniform_sampling_offset_size)

            #compute rgb
            rgb_fine, _, _ = self._compute_color(self.model_radiance,
                                                    t_fine_start, 
                                                    t_fine_end,
                                                    ray_origs,
                                                    ray_dirs,
                                                    pixel_width,
                                                    batch_size,
                                                    self.samples_per_ray_radiance)
            rgb_coarse = None

        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb_fine, rgb_coarse



    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        # unpack batch
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
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, pixel_width)


        # compute the loss
        loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:, 0])
        psnr = self.compute_psnr(loss)
        # Log metrics
        logs  = {f"{purpose}_loss_fine": loss,
                    f"{purpose}_psnr": psnr,
                    }

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:, 0])
            loss = loss + loss_coarse
            logs[f"{purpose}_loss_coarse"] = loss_coarse

        self.log_dict(logs)

        if loss.isnan():
            loss = th.tensor(1., requires_grad=True)
            warnings.warn("loss was nan - no optimization step performed")

        return loss



    def validation_transform_rays(self, ray_origs, ray_dirs, transform_params=None):
        return ray_origs, ray_dirs, transform_params



    def training_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, "train")


    def validation_step(self, batch, batch_idx):
        return self._step_helper(batch, batch_idx, "val")


    def configure_optimizers(self):
        # Create optimizer and schedular 

        optimizer = th.optim.Adam(
            [
                {
                    "params": param_group["parameters"], 
                    "lr": param_group["learning_rate_start"],
                    "weight_decay": param_group["weight_decay"],
                }
                for param_group in self.param_groups
            ],
            eps=1e-5
        )

        lr_scheduler = SchedulerLeNice(
            optimizer, 
            start_LR=[param_group["learning_rate_start"] for param_group in self.param_groups], 
            stop_LR= [param_group["learning_rate_stop"] for param_group in self.param_groups], 
            number_of_steps=[param_group["learning_rate_decay_end"] for param_group in self.param_groups],
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



    def compute_psnr(self, loss: th.Tensor) -> float:
        try:
            if loss <= 1e-7:
                print(f"WARN: Loss was {loss} - psnr not computed")
                return th.nan
            else:
                return -10 * math.log10(float(loss.cpu().detach().item()))
        except ValueError:
            print(f"WARN: Loss was {loss} and calculation crashes - psnr not computed")
            return th.nan