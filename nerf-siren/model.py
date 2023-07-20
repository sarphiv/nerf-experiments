import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput
from nerf_model import NerfModel


class NerfSiren(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray_fine: int,
        samples_per_ray_coarse: int,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.5,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        self.samples_per_ray_fine = samples_per_ray_fine
        self.samples_per_ray_coarse = samples_per_ray_coarse
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        
        self.weight_decay = weight_decay

        self.model_coarse = NerfModel(input_scale=32)
        self.model_fine = NerfModel(input_scale=32)


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

        # The below is just the code but in a more mathematical notation
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

    
    def _compute_color(self, model: NerfModel,
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
        rgb_coarse, weights, sample_dist_coarse = self._compute_color(
            self.model_coarse,
            t_coarse,
            ray_origs,
            ray_dirs,
            batch_size,
            self.samples_per_ray_coarse
        )
        

        ### Fine sampling
        # Sample t
        t_fine = self._sample_t_fine(t_coarse, weights, sample_dist_coarse)
        # compute rgb
        rgb_fine, _, _ = self._compute_color(
            self.model_fine,
            t_fine,
            ray_origs,
            ray_dirs,
            batch_size,
            self.samples_per_ray_coarse + self.samples_per_ray_fine
        )

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
        )
        scheduler = th.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=self.learning_rate_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


