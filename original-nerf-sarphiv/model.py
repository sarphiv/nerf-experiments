import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Callable, Iterator, Literal, cast

from data_module import DatasetOutput


# def arange(n: th.Tensor) -> th.Tensor:
#     out = []
#     i = 0
#     while i < n:
#         out.append(i)
#         i += 1
#     return th.tensor(out, device=n.device)


# def sample_fine_helper(t_start, t_stop, samples_per_bin):
#     # sample t within each bin
#     n = samples_per_bin + 1
#     interval_size = (t_stop - t_start) / n
#     t_fine = th.hstack([t_start[i] + arange(n[i])*interval_size[i] for i in range(len(n))])
#     return t_fine

# def sample_fine(t_start, t_stop, samples_per_bin):
#      batch_size = samples_per_bin.shape[0]
#      return th.stack([
#             sample_fine_helper(t_start, t_stop, samples_per_bin[i])
#             for i in range(batch_size)
#         ])
    

# sample_fine = th.vmap(sample_fine_helper, in_dims=(None, None, 0))

class FourierFeatures(nn.Module):
    def __init__(self, levels: int):
        super().__init__()

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


class NerfOriginalBase(nn.Module):
    """
    A base class for the original NeRF model.
    with one change: uses shifted softplus instead of ReLU.
    """
    def __init__(
        self,
        fourier_levels_pos: int,
        fourier_levels_dir: int,
        use_residual_color: bool = False,
    ):
        super().__init__()

        self.fourier_levels_pos = fourier_levels_pos
        self.fourier_levels_dir = fourier_levels_dir
        self.use_residual_color = use_residual_color

        self.model_fourier_pos = FourierFeatures(levels=self.fourier_levels_pos)
        self.model_fourier_dir = FourierFeatures(levels=self.fourier_levels_dir)

        self.model_density_1 = nn.Sequential(
            nn.Linear(3*2*self.fourier_levels_pos, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )
        self.model_density_2 = nn.Sequential(
            nn.Linear(256 + 3*2*self.fourier_levels_pos, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256+1 + 3*self.use_residual_color),
        )

        self.softplus = nn.Softplus(threshold=8)

        self.model_color = nn.Sequential(
            nn.Linear(256 + 3*2*self.fourier_levels_dir, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, pos: th.Tensor, dir: th.Tensor):
        fourier_pos = self.model_fourier_pos(pos)
        fourier_dir = self.model_fourier_dir(dir)
        
        z = self.model_density_1(fourier_pos)
        z = self.model_density_2(th.cat((z, fourier_pos), dim=1))

        # NOTE: Using shifted softplus like mip-NeRF instead of ReLU as in the original paper.
        #  The weight initialization seemed to cause negative initial values.

        density = self.softplus(z[:, 256] - 1)
        rgb = self.model_color(th.cat((z[:, :256], fourier_dir), dim=1))

        if self.use_residual_color:
            rgb_base = z[:, 257:260]
            rgb = self.sigmoid(rgb_base + rgb)
        else:
            rgb = self.sigmoid(rgb)

        return density, rgb

class NerfOriginal(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray_fine: int,
        samples_per_ray_coarse: int,
        fourier_levels_pos: int,
        fourier_levels_dir: int,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.5,
        weight_decay: float = 0.0,
        use_residual_color: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        self.samples_per_ray_fine = samples_per_ray_fine
        self.samples_per_ray_coarse = samples_per_ray_coarse
        
        self.fourier_levels_pos = fourier_levels_pos
        self.fourier_levels_dir = fourier_levels_dir
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        
        self.weight_decay = weight_decay
        
        self.use_residual_color = use_residual_color
        
        self.model_coarse = NerfOriginalBase(
            fourier_levels_pos=self.fourier_levels_pos,
            fourier_levels_dir=self.fourier_levels_dir,
            use_residual_color=self.use_residual_color
        )

        self.model_fine = NerfOriginalBase(
            fourier_levels_pos=self.fourier_levels_pos,
            fourier_levels_dir=self.fourier_levels_dir,
            use_residual_color=self.use_residual_color
        )

        # self.model_coarse = NerfOriginalCoarse(
        #     fourier_levels_pos=self.fourier_levels_pos
        # )
        # self.model_fine = NerfOriginalFine(
        #     fourier_levels_pos=self.fourier_levels_pos, 
        #     fourier_levels_dir=self.fourier_levels_dir
        # )

    
    def _sample_coarse(self, origins: th.Tensor, directions: th.Tensor):

        rays_n = origins.shape[0]


        ### Compute t 

        # Calculate interval size for each ray sample.
        #  Solve for interval in: near + interval*(samples+1) = far
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / self.samples_per_ray_coarse
        
        # Calculate direction scaling range for each ray,
        #  such that they start at the near sphere and end at the far sphere
        # NOTE: Subtracting one interval size to avoid sampling past far sphere
        
        t = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray_coarse, 
            device=self.device
        ).unsqueeze(0).repeat(rays_n, 1)
        
        # Perturb sample positions
        t += th.rand_like(t, device=self.device) * interval_size

        ### Calculate sample positions and directions
        # positions, directions, distances = self._compute_positions(origins, directions, t)
        positions = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)
        distances = th.hstack((
            t[:,1:] - t[:,:-1],
            self.far_sphere_normalized - t[:,-1:]
        ))
        directions = directions.unsqueeze(1).repeat(1, positions.shape[1], 1)
        return positions, directions, distances, t
    

    def _sample_fine(self, origins: th.Tensor, directions: th.Tensor, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor):

        ### Compute t
        weights = weights.squeeze(2)
        sample_idx = th.multinomial(weights, self.samples_per_ray_fine, replacement=True)
        t_fine = t_coarse.gather(1, sample_idx)
        t_fine += th.rand_like(t_fine, device=self.device) * distances_coarse.gather(1, sample_idx)
        t_fine = th.cat((t_coarse, t_fine), dim=1)
        t_fine = th.sort(t_fine, dim=1).values



        ### Calculate sample positions and directions
        # positions, directions, distances = self._compute_positions(origins, directions, t_fine)
        positions = origins.unsqueeze(1) + t_fine.unsqueeze(2) * directions.unsqueeze(1)
        distances = th.hstack((
            t_fine[:,1:] - t_fine[:,:-1],
            self.far_sphere_normalized - t_fine[:,-1:]
        ))
        directions = directions.unsqueeze(1).repeat(1, positions.shape[1], 1)
        return positions, directions, distances, t_fine


    # def _compute_positions(self, origins: th.Tensor, directions: th.Tensor, t: th.Tensor):

    #     rays_n = origins.shape[0]
    #     samples_n = t.shape[0]

    #     # Repeat origins and directions for each sample
    #     # origins = origins.repeat_interleave(samples_n, dim=0)
    #     # directions = directions.repeat_interleave(samples_n, dim=0)
        
    #     # Calculate sample positions
    #     # NOTE: using the same t for all samples - a bit hacky and not exactly the same as in the original paper
    #     # TODO: I don't like these unsqueezes
    #     positions = origins.unsqueeze(1) + t.unsqueeze(0).unsqueeze(2) * directions.unsqueeze(1)


    #     # Calculate distance between samples
    #     # NOTE: Direction vectors are unit vectors
    #     # t = t.view(rays_n, samples_n)
    #     distances = th.hstack((
    #         t[1:] - t[:-1],
    #         self.far_sphere_normalized - t[-1]
    #     ))

    #     # Group samples by ray
    #     # positions = positions.view(rays_n, samples_n, 3)
    #     # directions = directions.view(rays_n, samples_n, 3)

    #     # Return sample positions and directions with correct shape
    #     # return positions, directions.unsqueeze(1).expand(positions.shape), distances
    #     return positions, directions.unsqueeze(1).repeat(1, positions.shape[1], 1), distances


    def _render(self, densities: th.Tensor, colors: th.Tensor, distances: th.Tensor):
        """Densities and colors are shaped [batch_size, samples_per_ray, ...]
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



    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # Amount of pixels to render
        rays_n = ray_origs.shape[0]
        
        
        
        ### Coarse sampling
        # Sample positions and directions
        sample_pos_coarse, sample_dir_coarse, sample_dist_coarse, t_coarse = self._sample_coarse(ray_origs, ray_dirs)
        # Ungroup samples by ray
        sample_pos_coarse = sample_pos_coarse.view(rays_n * self.samples_per_ray_coarse, 3)
        sample_dir_coarse = sample_dir_coarse.view(rays_n * self.samples_per_ray_coarse, 3)
        # Evaluate density and color at sample positions
        sample_density_coarse, sample_rgb_coarse = self.model_coarse(sample_pos_coarse, sample_dir_coarse)
        # Group samples by ray
        sample_density_coarse = sample_density_coarse.view(rays_n, self.samples_per_ray_coarse)
        sample_color_coarse = sample_rgb_coarse.view(rays_n, self.samples_per_ray_coarse, 3)
        
        # Compute weights for fine sampling
        rgb_coarse, weights = self._render(sample_density_coarse, sample_color_coarse, sample_dist_coarse)
        

        ### Fine sampling
        # Sample positions and directions
        sample_pos_fine, sample_dir_fine, sample_dist_fine, t_fine = self._sample_fine(ray_origs, ray_dirs, t_coarse, weights, sample_dist_coarse)
        # Ungroup samples by ray

        spr = self.samples_per_ray_fine + self.samples_per_ray_coarse

        sample_pos_fine = sample_pos_fine.view(rays_n * spr, 3)
        sample_dir_fine = sample_dir_fine.view(rays_n * spr, 3)
        # Evaluate density and color at sample positions
        sample_density_fine, sample_rgb_fine = self.model_fine(sample_pos_fine, sample_dir_fine)
        # Group samples by ray
        sample_density_fine = sample_density_fine.view(rays_n, spr)
        sample_color_fine = sample_rgb_fine.view(rays_n, spr, 3)

        # Compute color for each pixel
        rgb_fine, _ = self._render(sample_density_fine, sample_color_fine, sample_dist_fine)


        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb_fine, rgb_coarse


    def _step_helpher(self, batch: DatasetOutput, batch_idx: int, stage: str):
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
            weight_decay=self.weight_decay
        )
        scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.learning_rate_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
