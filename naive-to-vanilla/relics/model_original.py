import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput

MAGIC_NUMBER = 7

class FourierFeatures(nn.Module):
    def __init__(self, levels: int, scale: float = 2*th.pi): 
        """
        Positional encoding using Fourier features of "levels" periods. 
        
        """
        super().__init__()

        self.levels = levels
        self.scale = scale


    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for j in [0, levels-1]
        """
        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(x.shape[1])
        args = x.repeat_interleave(self.levels, dim=1) * scale

        return th.hstack((th.cos(args), th.sin(args)))


class NerfModel(nn.Module):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
        fourier_levels_pos: int,
        fourier_levels_dir: int
    ):
        """
        An instance of a NeRF model the architecture; 
        
            Hn( Hn( Ep(x)), Ep(x))          -> density 
        H(  Hn( Hn( Ep(x)), Ep(x)), Ed(d))  -> color 

        H: a hidden fully connected layer with hidden_dim neurons,
        Hn: H applied n_hidden times, 
        Ep: is the positional encoding with fourier_levels_pos levels,
        Ed: is the direction encoding with fourier_levels_dir levels,
        x: is the position
        d: is the direction of the camera ray
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.position_encoder = FourierFeatures(fourier_levels_pos, 2*th.pi)
        self.direction_encoder = FourierFeatures(fourier_levels_dir, 1.0)

        # Creates the first module of the network 
        self.model_density_1 = self.contruct_model_density(fourier_levels_pos*2*3,
                                                           self.hidden_dim,
                                                           self.hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Creates the second module of the network (feeds the positional incodings into the network again)
        self.model_density_2 = self.contruct_model_density(self.hidden_dim + fourier_levels_pos*2*3,
                                                           self.hidden_dim,
                                                           self.hidden_dim + 1)
        self.softplus = nn.Softplus(threshold=8)

        # Creates the final layer of the model that outputs the color
        self.model_color = nn.Sequential(
            nn.Linear(self.hidden_dim + fourier_levels_dir*2*3, self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim//2, 3)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # Ep(x), Ed(x)
        pos = self.position_encoder(pos)
        dir = self.direction_encoder(dir)
        
        # Hn(Hn(Ep(x)), Ep(x))
        z = self.model_density_1(pos)
        z = self.relu(z)
        z = self.model_density_2(th.cat((z, pos), dim=1))

        # NOTE: Using shifted softplus like mip-NeRF instead of ReLU as in the original paper.
        #  The weight initialization seemed to cause negative initial values.

        density = self.softplus(z[:, self.hidden_dim] - 1)

        # H(Hn(Hn(Ep(x)), Ep(x)), Ed(d))
        # Extract the color estimate
        rgb = self.model_color(th.cat((z[:, :self.hidden_dim], dir), dim=1))

        # Clamp the rgb values to be between 0 and 1
        rgb = self.sigmoid(rgb)

        return density, rgb

    def contruct_model_density(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """
        Creates a FFNN with n_hidden layers and fits the input and output dimensionality.
        The activation function is ReLU 
        """
        if self.n_hidden == 0:
            return nn.Linear(input_dim, output_dim)
        else:
            # The initial and final layers have special dimensions
            layer1 = nn.Linear(input_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, output_dim)
            intermediate_layers = []
            
            # Create n_hidden identical layers 
            for _ in range(self.n_hidden-1):
                intermediate_layers += [nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim)]

            # Concatenate the layers into one sequential
            return nn.Sequential(layer1, *intermediate_layers, nn.ReLU(True), layer2)


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
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # The near and far sphere distances 
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized

        # Samples per ray for coarse and fine sampling
        self.samples_per_ray_fine = samples_per_ray_fine
        self.samples_per_ray_coarse = samples_per_ray_coarse
        
        # Hyper parameters for the optimizer 
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        
        # Coarse model -> only job is to find out where to sample more by predicting the density
        self.model_coarse = NerfModel(
            n_hidden=4,
            hidden_dim=256,
            fourier_levels_pos=fourier_levels_pos,
            fourier_levels_dir=fourier_levels_dir
        )

        # Fine model -> actual prediction, samples more densely
        self.model_fine = NerfModel(
            n_hidden=4,
            hidden_dim=256,
            fourier_levels_pos=fourier_levels_pos,
            fourier_levels_dir=fourier_levels_dir
        )

    def _get_intervals(self, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        From t-values get the t corresponding to the start and end of the bins.
        
        Parameters:
        -----------
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.
            
        Returns:
        --------
            t_start: Tensor of shape (batch_size, samples_per_ray) - the t value for the begining of the bin
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
        t_coarse += th.rand_like(t_coarse, device=self.device) * interval_size

        return self._get_intervals(t_coarse)
    

    def _sample_t_fine(self, t_coarse: th.Tensor, weights: th.Tensor, distances_coarse: th.Tensor) ->  tuple[th.Tensor, th.Tensor]:
        """
        Using the coarsely sampled t values to decide where to sample more densely. 
        The weights express how large a percentage of the light is blocked by that specific segment, hence that percentage of the new samples should be in that segment.

        Parameters:
        -----------
            t_coarse:           Tensor of shape (batch_size, samples_per_ray_coarse) - the t-values for the beginning of each bin
            weights:            Tensor of shape (batch_size, samples_per_ray_coarse) (these are assumed to almost sum to 1)
            distances_coarse:   Tensor of shape (batch_size, samples_per_ray_coarse)
        
        Returns:
        --------
            t_start:            Tensor of shape (batch_size, samples_per_ray) - start of the t-bins
            t_end:              Tensor of shape (batch_size, samples_per_ray) - end of the t-bins
        
        """
        # Initialization 
        batch_size = t_coarse.shape[0]
        device = t_coarse.device

        # Each segment needs weight*samples_per_ray_fine new samples plus 1 because of the coarse sample 
        fine_samples = th.round(weights*self.samples_per_ray_fine)
        # Rounding might cause the sum to be less than or larger than samples_per_ray_fine, so we add the difference to the largest segment (especially since weights don't sum all the way to 1)
        fine_samples[th.arange(batch_size), th.argmax(fine_samples, dim=1)] += self.samples_per_ray_fine - fine_samples.sum(dim=1)
        fine_samples += 1
        fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1, device=device), fine_samples.cumsum(dim=1)))
        
        # Instanciate the t_fine tensor and arange is used to mask the correct t values for each batch
        arange = th.arange(self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device).unsqueeze(0)
        t_fine = th.zeros(batch_size, self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device)

        for i in range(self.samples_per_ray_coarse):
            # Pick out the samples for each segment, everything within cumsum[i] and cumsum[i+1] is in segment i because the difference is the new samples
            # This mask is for each ray in the batch 
            mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
            # Set samples to be the coarsely sampled point 
            t_fine += t_coarse[:, i].unsqueeze(-1)*mask
            # And spread them out evenly in the segment by deviding the length of the segment with the amount of samples in that segment
            t_fine += (arange - fine_samples_cum_sum[:, i].unsqueeze(-1))*mask*distances_coarse[:, i].unsqueeze(-1)/fine_samples[:, i].unsqueeze(-1)

        return self._get_intervals(t_fine)


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

        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb_fine, rgb_coarse


    ############ pytorch lightning functions ############

    def _step_helpher(self, batch: DatasetOutput, batch_idx: int, stage: str):
        """
        general function for training and validation step
        """
        ray_origs, ray_dirs, ray_colors = batch
        
        # Compute the rgb values for the given rays for both models
        ray_colors_pred_coarse, ray_colors_pred_fine = self(ray_origs, ray_dirs)

        # Compute the individual losses
        proposal_loss = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors)
        radiance_loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors)

        # Calculate total loss
        network_loss = proposal_loss + radiance_loss
        
        # Calculate PSNR
        # NOTE: Cannot calculate SSIM because it relies on image patches
        # NOTE: Cannot calculate LPIPS because it relies on image patches
        psnr = -10 * th.log10(radiance_loss)

        # Log metrics
        self.log_dict({
            f"{stage}_network_loss": network_loss,
            f"{stage}_proposal_loss": proposal_loss,
            f"{stage}_radiance_loss": radiance_loss,
            f"{stage}_psnr": psnr,
        }) 

        return network_loss

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


