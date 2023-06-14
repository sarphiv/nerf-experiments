import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput



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



class NerfOriginalCoarse(nn.Module):
    def __init__(
        self,
        fourier_levels_pos: int
    ):
        super().__init__()

        self.fourier_levels_pos = fourier_levels_pos


        self.model_fourier_pos = FourierFeatures(levels=self.fourier_levels_pos)

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
            nn.Linear(256, 1),
            nn.ReLU(inplace=True)
        )


    def forward(self, pos: th.Tensor):
        fourier_pos = self.model_fourier_pos(pos)
        
        z = self.model_density_1(fourier_pos)
        z = self.model_density_2(th.cat((z, fourier_pos), dim=1))

        # Return density
        return z



class NerfOriginalFine(nn.Module):
    def __init__(
        self,
        fourier_levels_pos: int,
        fourier_levels_dir: int,
    ):
        super().__init__()

        self.fourier_levels_pos = fourier_levels_pos
        self.fourier_levels_dir = fourier_levels_dir


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
            nn.Linear(256, 256+1),
        )
        
        self.model_color = nn.Sequential(
            nn.Linear(256 + 3*2*self.fourier_levels_dir, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )


    def forward(self, pos: th.Tensor, dir: th.Tensor):
        fourier_pos = self.model_fourier_pos(pos)
        fourier_dir = self.model_fourier_dir(dir)
        
        z = self.model_density_1(fourier_pos)
        z = self.model_density_2(th.cat((z, fourier_pos), dim=1))

        # NOTE: Using exp instead of ReLU as in the original paper.
        #  The weight initialization seemed to cause negative initial values.
        density = th.exp(z[:, 256])
        rgb = self.model_color(th.cat((z[:, :256], fourier_dir), dim=1))

        return density, rgb



class NerfOriginal(pl.LightningModule):
    def __init__(
        self, 
        width: int, 
        height: int, 
        focal_length: float,
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray: int,
        fourier_levels_pos: int,
        fourier_levels_dir: int,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.5,
        learning_rate_decay_patience: int = 80,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.width = width
        self.height = height
        
        self.focal_length = focal_length
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        self.samples_per_ray = samples_per_ray
        
        self.fourier_levels_pos = fourier_levels_pos
        self.fourier_levels_dir = fourier_levels_dir
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_patience = learning_rate_decay_patience
        
        self.weight_decay = weight_decay
        
        
        # self.model_coarse = NerfOriginalCoarse(
        #     fourier_levels_pos=self.fourier_levels_pos
        # )
        self.model_fine = NerfOriginalFine(
            fourier_levels_pos=self.fourier_levels_pos, 
            fourier_levels_dir=self.fourier_levels_dir
        )



    def _get_sample_positions(self, origins: th.Tensor, directions: th.Tensor):
        # Amount of rays
        rays_n = origins.shape[0]
        
        # Calculate interval size for each ray sample.
        #  Solve for interval in: near + interval*(samples+1) = far
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / (self.samples_per_ray + 1)
        
        # Calculate direction scaling range for each ray,
        #  such that they start at the near sphere and end at the far sphere
        # NOTE: Subtracting one interval size to avoid sampling past far sphere
        t = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray, 
            device=self.device
        ).unsqueeze(1).repeat(rays_n, 1)
        
        # Perturb sample positions
        t += th.rand_like(t, device=self.device) * interval_size
        
        # Repeat origins and directions for each sample
        origins = origins.repeat_interleave(self.samples_per_ray, dim=0)
        directions = directions.repeat_interleave(self.samples_per_ray, dim=0)
        
        # Calculate sample positions
        positions = origins + t * directions


        # Calculate distance between samples
        # NOTE: Direction vectors are normalized
        t = t.view(rays_n, self.samples_per_ray)
        distances = th.hstack((
            t[:, 1:] - t[:, :-1],
            self.far_sphere_normalized - t[:, -1:]
        ))


        # Group samples by ray
        positions = positions.view(rays_n, self.samples_per_ray, 3)
        directions = directions.view(rays_n, self.samples_per_ray, 3)


        # Return sample positions and directions with correct shape
        return positions, directions, distances


    def _render(self, densities: th.Tensor, colors: th.Tensor, distances: th.Tensor):
        """Densities and colors are shaped [batch_size, samples_per_ray, ...]
        """
        blocking_neg = (-densities * distances).unsqueeze(-1)
        alpha = 1 - th.exp(blocking_neg)
        alpha_int = th.hstack((
            th.ones((blocking_neg.shape[0], 1, 1), device=self.device),
            th.exp(th.cumsum(blocking_neg[:, :-1], dim=1))
        ))

        return th.sum(alpha_int*alpha*colors, dim=1)



    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor):
        # Amount of pixels to render
        rays_n = ray_origs.shape[0]
        
        
        
        
        
        
        # TODO: Implement hiarchical sampling
        
        
        
        
        
        
        
        # Retrieve sample positions and view directions
        sample_pos_fine, sample_dir_fine, sample_dist_fine = self._get_sample_positions(ray_origs, ray_dirs)

        # Ungroup samples by ray
        sample_pos_fine = sample_pos_fine.view(rays_n * self.samples_per_ray, 3)
        sample_dir_fine = sample_dir_fine.view(rays_n * self.samples_per_ray, 3)


        # Evaluate density and color at sample positions
        sample_density_fine, sample_rgb_fine = self.model_fine(sample_pos_fine, sample_dir_fine)


        # Group samples by ray
        sample_density_fine = sample_density_fine.view(rays_n, self.samples_per_ray)
        sample_color_fine = sample_rgb_fine.view(rays_n, self.samples_per_ray, 3)


        # Compute color for each pixel
        rgb = self._render(sample_density_fine, sample_color_fine, sample_dist_fine)


        # Return colors for the given pixel coordinates
        return rgb



    def training_step(self, batch: DatasetOutput, batch_idx: int):
        ray_origs, ray_dirs, ray_colors = batch
        
        ray_colors_pred = self(ray_origs, ray_dirs)

        loss = nn.functional.mse_loss(ray_colors_pred, ray_colors)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch: DatasetOutput, batch_idx: int):
        ray_origs, ray_dirs, ray_colors = batch
        
        ray_colors_pred = self(ray_origs, ray_dirs)

        loss = nn.functional.mse_loss(ray_colors_pred, ray_colors)
        self.log("val_loss", loss)

        return loss


    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=self.learning_rate_decay, 
            patience=self.learning_rate_decay_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }
