import torch as th
import torch.nn as nn
import pytorch_lightning as pl



class FourierFeatures(nn.Module):
    def __init__(self, levels: int):
        super().__init__()

        self.levels = levels


    def forward(self, x: th.Tensor):
        scale = (2**th.arange(self.levels) * th.pi) \
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
        
        density = th.relu(z[:, 256])
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
        rays_per_image: int,
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
        
        self.rays_per_image = rays_per_image
        self.samples_per_ray = samples_per_ray
        
        self.fourier_levels_pos = fourier_levels_pos
        self.fourier_levels_dir = fourier_levels_dir
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_patience = learning_rate_decay_patience
        
        self.weight_decay = weight_decay
        
        
        self.model_coarse = NerfOriginalCoarse(
            fourier_levels_pos=self.fourier_levels_pos
        )
        self.model_fine = NerfOriginalFine(
            fourier_levels_pos=self.fourier_levels_pos, 
            fourier_levels_dir=self.fourier_levels_dir
        )



    def _get_random_pixels(self):
        pixel_coords = th.rand((self.rays_per_image, 2), device=self.device)
        pixel_coords[:, 0] *= self.width
        pixel_coords[:, 1] *= self.height

        return pixel_coords.int()

    
    def _get_rays(self, pixel_coords: th.Tensor, camera_to_world: th.Tensor):
        # Conversion from pixel coordinates to camera coordinates
        image_size = th.tensor([self.width, self.height], device=self.device)
        
        # NOTE: Normalized such that z=-1 via the focal length.
        #  Camera is looking in the negative z direction.
        
        
        
        
        
        # WARN: This matrix multiplication may be wrong
        directions = (camera_to_world[:3, :3] @ th.hstack((
            (pixel_coords/image_size - 0.5) / self.focal_length, 
            -th.ones((pixel_coords.shape[0], 1), device=self.device)
        )).T).T
        # directions = (camera_to_world[:3, :3].T @ th.hstack((
        #     (pixel_coords/image_size - 0.5) / self.focal_length, 
        #     -th.ones((pixel_coords.shape[0], 1), device=self.device)
        # )).T).T
        # directions = (camera_to_world[:3, :3].T @ th.hstack((
        #     (pixel_coords/image_size - 0.5) / self.focal_length, 
        #     th.ones((pixel_coords.shape[0], 1), device=self.device)
        # )).T).T
        # directions = (th.hstack((
        #     (pixel_coords/image_size - 0.5) / self.focal_length, 
        #     -th.ones((pixel_coords.shape[0], 1), device=self.device)
        # )) @ camera_to_world[:3, :3].T)
        # x, y = th.meshgrid(
        #     th.linspace(-0.5, 0.5, self.width, device=self.device) / self.focal_length,
        #     th.linspace(-0.5, 0.5, self.height, device=self.device) / self.focal_length,
        #     indexing="xy"
        # )
        # directions = (th.stack((x, y, th.ones_like(x, device=self.device)), dim=-1) @ camera_to_world[:3, :3].T).view(-1, 3)
        # directions = directions[th.randint(0, directions.shape[0], (pixel_coords.shape[0],))]









        # Normalizing directions again such that distance calculations are simpler and faster
        directions /= th.norm(directions, dim=1, p=2, keepdim=True)
        

        # Retrieve ray origin and repeat it for each ray
        origins = camera_to_world[:3, 3].expand(pixel_coords.shape[0], 3)


        return origins, directions


    def _get_sample_positions(self, origins: th.Tensor, directions: th.Tensor):
        # Amount of rays
        rays_n = origins.shape[0]
        
        # Calculate interval size for each ray sample
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / self.samples_per_ray
        
        # Calculate direction scaling range for each ray,
        #  such that they start at the near plane and end at the far plane
        # NOTE: Subtracting one interval size to avoid sampling past far plane.
        t = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray, 
            device=self.device
        ).unsqueeze(1).repeat(rays_n, 1)
        
        # Perturb sample positions
        t += th.rand_like(t, device=self.device) * interval_size
        
        # Repeat origins and directions for each sample
        origins = origins.repeat(self.samples_per_ray, 1)
        directions = directions.repeat(self.samples_per_ray, 1)
        
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



    def forward(self, pixel_coords: th.Tensor, camera_to_world: th.Tensor):
        # Amount of pixels to render
        pixels_n = pixel_coords.shape[0]

        # Get ray origins and directions from pixel coordinates and projection matrix
        ray_origs, ray_dirs  = self._get_rays(pixel_coords, camera_to_world)
        
        
        
        
        
        
        # TODO: Implement hiarchical sampling
        
        
        
        
        
        # Retrieve sample positions and view directions
        sample_pos_fine, sample_dir_fine, sample_dist_fine = self._get_sample_positions(ray_origs, ray_dirs)

        # Ungroup samples by ray
        sample_pos_fine = sample_pos_fine.view(pixels_n * self.samples_per_ray, 3)
        sample_dir_fine = sample_dir_fine.view(pixels_n * self.samples_per_ray, 3)


        # Evaluate density and color at sample positions
        sample_density_fine, sample_rgb_fine = self.model_fine(sample_pos_fine, sample_dir_fine)


        # Group samples by ray
        sample_density_fine = sample_density_fine.view(pixels_n, self.samples_per_ray)
        sample_color_fine = sample_rgb_fine.view(pixels_n, self.samples_per_ray, 3)


        # Compute color for each pixel
        rgb = self._render(sample_density_fine, sample_color_fine, sample_dist_fine)


        # Return colors for the given pixel coordinates
        return rgb



    def training_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int):
        # NOTE: Batch size is 1
        camera_to_world, image = batch
        camera_to_world, image = camera_to_world[0], image[0]

        pixel_coords = self._get_random_pixels()
        
        color = image[:3, pixel_coords[:, 1], pixel_coords[:, 0]].permute(1, 0)
        color_pred = self(pixel_coords.float(), camera_to_world)

        loss = nn.functional.mse_loss(color_pred, color)
        self.log("train_loss", loss)

        return loss


    def validation_step(self, batch, batch_idx):
        # NOTE: Batch size is 1
        camera_to_world, image = batch
        camera_to_world, image = camera_to_world[0], image[0]

        pixel_coords = self._get_random_pixels()
        
        color = image[:3, pixel_coords[:, 1], pixel_coords[:, 0]].permute(1, 0)
        color_pred = self(pixel_coords.float(), camera_to_world)

        loss = nn.functional.mse_loss(color_pred, color)
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
