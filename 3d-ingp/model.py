import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Callable, Iterator, Literal, cast

from data_module import DatasetOutput
import math

import numpy as np
import os



class INGPTable(nn.Module):
    def __init__(self, resolution, table_size, n_features, pi1, pi2, pi3):
        super().__init__()
        self.resolution = resolution
        self.table_size = table_size
        self.n_features = n_features
        self.pi1 = pi1
        self.pi2 = pi2
        self.pi3 = pi3

        self.bijective = table_size >= (resolution + 1)**3

        if self.bijective:
            self.table = nn.Parameter(
                (th.rand(((resolution+1)**3, n_features))*2 - 1)*10**(-4)
                )
        else:
            self.table = nn.Parameter(
                (th.rand((table_size, n_features))*2 - 1)*10**(-4)
                )
    
    def compute_idx(self, corners: th.Tensor):

        if self.bijective:
            corners = corners.clip(0, self.resolution)
            idx = th.sum(corners * th.tensor([1, self.resolution+1, (self.resolution+1)**2], device=corners.device), dim=2)
        else:
            idx = self.hash(corners)
        return idx

    def hash(self, x):
        # x: (batch_size, 2**d, d) - d=3 (we are gonna have d 2's)
        # output: (batch_size, 2**d)

        y1 = self.pi1 * x[...,0]
        y2 = self.pi2 * x[...,1]
        y3 = self.pi3 * x[...,2]

        y = th.bitwise_xor(y1, y2)
        y = th.bitwise_xor(y, y3)
        y = th.remainder(y, self.table_size)
        
        return y

    def forward(self, x: th.Tensor):
        # x: (batch_size, data_dim)
        # output: (batch_size, n_features)

        batch_size, data_dim = x.shape

        # get corners:
        x_scaled = x * self.resolution
        x_floor = th.floor(x_scaled)
        x_ceil = x_floor + 1
        x_lim = th.stack((x_floor, x_ceil), dim=1)

        idx_list = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]

        corners = th.stack(
            [
                x_lim[:,[i,j,k], th.arange(3)] for i,j,k in idx_list
              ],
        dim=1).to(th.int64)


        feature_idx = self.compute_idx(corners)
        features = self.table[feature_idx]

        # get weights:
        x_diff = x_scaled.unsqueeze(1) - corners
        x_diff = th.abs(x_diff)
        weights = 1 - x_diff
        weights = th.prod(weights, dim=-1)

        # get output:
        output = th.sum(features * weights.unsqueeze(-1), dim=1)
        return output

class INGPEncoding(nn.Module):
    def __init__(self, resolution_max, resolution_min,
                 table_size, n_features, n_levels,
                 pi1=1, pi2=2654435761, pi3=805459861):
        super().__init__()
        self.output_dim = n_features*n_levels
        self.resolution_max = resolution_max
        self.resolution_min = resolution_min
        self.table_size = table_size
        self.n_features = n_features
        self.n_levels = n_levels
        self.b = 1 if n_levels==1 else math.exp((math.log(resolution_max) - math.log(resolution_min)) / (n_levels-1))

        self.resolution = th.floor(resolution_min * self.b**th.arange(n_levels))

        self.encodings = nn.ModuleList(
            [INGPTable(int(r), table_size, n_features, pi1, pi2, pi3) for r in self.resolution]
        )
    
    def forward(self, x):
        # x: (batch_size, data_dim)
        # output: (batch_size, n_features*n_levels)

        # NOTE: hardcoding the normalization to be in the cube (0,1)^3

        x_norm = x / 8 + 0.5

        output = th.cat([enc(x_norm) for enc in self.encodings], dim=1)

        return output


class FourierFeatures(nn.Module):
    def __init__(self, levels: int):
        """
        Positional encoding using Fourier features.
        
        """
        super().__init__()
            
        self.output_dim = 3*2*levels

        self.levels = levels


    def forward(self, x: th.Tensor):
        # NOTE: Not multiplying with pi as in the original paper,
        #  because their own implementation does not do this.
        scale = (2**th.arange(self.levels)) \
            .repeat(x.shape[1]) \
            .to(x.device)
        args = x.repeat_interleave(self.levels, dim=1) * scale

        # NOTE: Sines and cosines on the same level are not adjacent 
        #  as in the original paper. Network should be invariant to this,
        #  so there should be no loss difference. Computation is faster with this.
        return th.hstack((th.cos(args), th.sin(args)))


class NerfModelINGP(nn.Module):
    def __init__(self, n_hidden: int,
                 hidden_dim: int,
                 position_encoder: INGPEncoding,
                 direction_encoder: FourierFeatures,
                 ):
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder
        self.direction_encoder = direction_encoder


        if self.n_hidden == 0:
            self.model_density = nn.Linear(position_encoder.output_dim, hidden_dim + 1)
        else:
            layer1 = nn.Linear(position_encoder.output_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, hidden_dim + 1)
            intermediate_layers = []
            for _ in range(self.n_hidden-1):
                intermediate_layers += [nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim)]
            self.model_density = nn.Sequential(layer1, *intermediate_layers, nn.ReLU(True), layer2)
     
        self.softplus = nn.Softplus(threshold=8)

        self.model_color = nn.Sequential(
            nn.Linear(self.hidden_dim + self.direction_encoder.output_dim, self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim//2, 3)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, pos: th.Tensor, dir: th.Tensor):
        pos = self.position_encoder(pos)
        dir = self.direction_encoder(dir)
        
        z = self.model_density(pos)
        density = self.softplus(z[:, self.hidden_dim] - 1)
        rgb = self.model_color(th.cat((z[:, :self.hidden_dim], dir), dim=1))
        rgb = self.sigmoid(rgb)
        
        return density, rgb

class NaiveINGP(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray_fine: int,
        samples_per_ray_coarse: int,
        position_encoder: INGPEncoding,
        direction_encoder: FourierFeatures,
        n_hidden: int,
        hidden_dim: int,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.5,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        # self.save_hyperparameters()

        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        self.samples_per_ray_fine = samples_per_ray_fine
        self.samples_per_ray_coarse = samples_per_ray_coarse
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        
        self.weight_decay = weight_decay
        
        self.model_coarse = NerfModelINGP(
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            position_encoder=position_encoder,
            direction_encoder=direction_encoder,
        )
        self.model_fine = NerfModelINGP(
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            position_encoder=position_encoder,
            direction_encoder=direction_encoder,
        )
    
    def _sample_t_coarse(self, batch_size):
        """
        Sample t for coarse sampling.

        Returns:
        --------
            t_coarse: Tensor of shape (batch_size, samples_per_ray)
        
        """

        ### Compute t 

        # Calculate interval size for each ray sample.
        #  Solve for interval in: near + interval*(samples+1) = far
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / self.samples_per_ray_coarse
        
        # Calculate direction scaling range for each ray,
        #  such that they start at the near sphere and end at the far sphere
        # NOTE: Subtracting one interval size to avoid sampling past far sphere
        
        t_coarse = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray_coarse, 
            device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)
        
        # Perturb sample positions
        t_coarse += th.rand_like(t_coarse, device=self.device) * interval_size

        return t_coarse
    

    def _sample_t_fine(self, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor, linspace=True) -> th.Tensor:
        """
        Sample t using hierarchical sampling based on the weights from the coarse model.

        Parameters:
        -----------
            t_coarse: Tensor of shape (batch_size, samples_per_ray_coarse)
            weights: Tensor of shape (batch_size, samples_per_ray_coarse)
            distances_coarse: Tensor of shape (batch_size, samples_per_ray_coarse)
            linspace: bool - whether to use linspace instead of multinomial sampling
        
        Returns:
        --------
            t_fine: Tensor of shape (batch_size, samples_per_ray_coarse + samples_per_ray_fine) (so it contains the t_coarse sample points as well)
        
        """

        ### Compute t
        weights = weights.squeeze(2)
        batch_size = t_coarse.shape[0]
        device = t_coarse.device

        if linspace:
            fine_samples = th.round(weights*self.samples_per_ray_fine)
            fine_samples[th.arange(batch_size), th.argmax(fine_samples, dim=1)] += self.samples_per_ray_fine - fine_samples.sum(dim=1)
            fine_samples += 1
            fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1, device=device), fine_samples.cumsum(dim=1)))
            
            arange = th.arange(self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device).unsqueeze(0)
            t_fine = th.zeros(batch_size, self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device)

            for i in range(self.samples_per_ray_coarse):
                mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
                t_fine += t_coarse[:, i].unsqueeze(-1)*mask
                t_fine += (arange - fine_samples_cum_sum[:, i].unsqueeze(-1))*mask*distances_coarse[:, i].unsqueeze(-1)/fine_samples[:, i].unsqueeze(-1)

        else:

            sample_idx = th.multinomial(weights, self.samples_per_ray_fine, replacement=True)
            t_fine = t_coarse.gather(1, sample_idx)
            t_fine += th.rand_like(t_fine, device=self.device) * distances_coarse.gather(1, sample_idx)
            t_fine = th.cat((t_coarse, t_fine), dim=1)
            t_fine = th.sort(t_fine, dim=1).values

        return t_fine


    def _compute_positions(self, origins: th.Tensor, directions: th.Tensor, t: th.Tensor):
        """
        Compute the positions, distances and directions for the given rays and t values.

        Parameters:
        -----------
            origins: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            directions: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.


        Returns:
        --------
            positions: Tensor of shape (batch_size, samples_per_ray, 3) - the positions for the given t values
            distances: Tensor of shape (batch_size, samples_per_ray) - the distances between the positions - used for the weights
                        Appended is the distance between the last position and the far sphere.
            directions: Tensor of shape (batch_size, samples_per_ray, 3) - the directions for the given t values
        """


        positions = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)
        distances = th.hstack((
            t[:,1:] - t[:,:-1],
            self.far_sphere_normalized - t[:,-1:]
        ))
        directions = directions.unsqueeze(1).repeat(1, positions.shape[1], 1)
        return positions, directions, distances



    def _render_rays(self, densities: th.Tensor, colors: th.Tensor, distances: th.Tensor):
        """
        Render the rays using the given densities and colors.
        I.e. this is just an implementation of equation 3.

        Parameters:
        -----------
            densities: Tensor of shape (batch_size, samples_per_ray) - the densities for the given samples (sigma_j in the paper)
            colors: Tensor of shape (batch_size, samples_per_ray, 3) - the colors for the given samples (c_j in the paper)
            distances: Tensor of shape (batch_size, samples_per_ray) - the distances between the samples (delta_j in the paper)
        
        
        Returns:
        --------
            rgb: Tensor of shape (batch_size, 3) - the rgb values for the given rays
            weights: Tensor of shape (batch_size, samples_per_ray) - the weights used for the fine sampling

        """

        # blocking_neg = -sigma_j * delta_j
        # alpha_j = 1 - exp(blocking_neg)
        # alpha_int_i = T_i = exp(sum_{j=1}^{i-1} blocking_neg_j)
        # output = sum_{j=1}^N alpha_int_j * alpha_j * c_j (c_j = color_j)

        blocking_neg = (-densities * distances).unsqueeze(-1)
        alpha = 1 - th.exp(blocking_neg)
        alpha_int = th.hstack((
            th.ones((blocking_neg.shape[0], 1, 1), device=self.device),
            th.exp(th.cumsum(blocking_neg[:, :-1, :], dim=1))
        ))

        weights = alpha_int * alpha

        return th.sum(weights*colors, dim=1), weights

    
    def _compute_color(self, model: NerfModelINGP,
                            t: th.Tensor,
                            ray_origs: th.Tensor,
                            ray_dirs: th.Tensor,
                            batch_size: int,
                            samples_per_ray: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        """
        A helper function to compute the color for the given model, t values, ray origins and ray directions.
        Is just a wrapper around _compute_positions, model_base.forward and _render_rays.

        Parameters:
        -----------
            model: The model to use for computing the color (either the coarse or fine model)
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for. (output of _sample_t_coarse or _sample_t_fine)
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


        sample_pos, sample_dir, sample_dist  = self._compute_positions(ray_origs, ray_dirs, t)
        # Evaluate density and color at sample positions
        # Ungroup samples by ray
        sample_pos = sample_pos.view(batch_size * samples_per_ray, 3)
        sample_dir = sample_dir.view(batch_size * samples_per_ray, 3)
        # Evaluate density and color at sample positions
        sample_density, sample_color = model(sample_pos, sample_dir)
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
        t_coarse = self._sample_t_coarse(batch_size)

        #compute rgb
        rgb_coarse, weights, sample_dist_coarse = self._compute_color(self.model_coarse,
                                                t_coarse,
                                                ray_origs,
                                                ray_dirs,
                                                batch_size,
                                                self.samples_per_ray_coarse)
        

        ### Fine sampling
        # Sample t
        t_fine = self._sample_t_fine(t_coarse, weights, sample_dist_coarse)
        # compute rgb
        rgb_fine, _, _ = self._compute_color(self.model_fine,
                                                t_fine,
                                                ray_origs,
                                                ray_dirs,
                                                batch_size,
                                                self.samples_per_ray_coarse + self.samples_per_ray_fine)

        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb_fine, rgb_coarse


    ############ pytorch lightning functions ############

    def _step_helpher(self, batch: DatasetOutput, batch_idx: int, stage: str):
        """
        general function for training and validation step
        """
        ray_origs, ray_dirs, ray_colors = batch
        
        ray_colors_pred_coarse, ray_colors_pred_fine = self(ray_origs, ray_dirs)

        loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors)
        loss_fine = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors)
        loss = loss_coarse + loss_fine
        
        self.log(f"{stage}_loss", loss)

        return loss

    def training_step(self, batch: DatasetOutput, batch_idx: int):
        return self._step_helpher(batch, batch_idx, "train")

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        return self._step_helpher(batch, batch_idx, "val")


    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.99),
            eps=1e-15
        )
        # scheduler = th.optim.lr_scheduler.ExponentialLR(
        #     optimizer, 
        #     gamma=self.learning_rate_decay
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
        }


