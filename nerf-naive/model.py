import torch as th
import torch.nn as nn
import pytorch_lightning as pl

from data_module import DatasetOutput

MAGIC_NUMBER = 7

class NerfModel(nn.Module):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int
    ):
        """
        An instance of a naive NeRF model architecture; 
        6 inputs (position and direction) 
        n_hidden layers with hidden_dim neurons each
        4 outputs (rgb and density)
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        # Define the naive model 
        self.model = self.contruct_model(6,
                                        self.hidden_dim,
                                        4)
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus(threshold=8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # The neural network.
        rgbd = self.model(th.cat((pos, dir), dim=1))

        # Clamp the rgb values to be between 0 and 1
        rgb = self.sigmoid(rgbd[:, :3])

        # Softplus the density 
        density = self.softplus(rgbd[:, 3])

        return density, rgb

    def contruct_model(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
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
                intermediate_layers += [self.relu, nn.Linear(hidden_dim, hidden_dim)]

            # Concatenate the layers into one sequential
            return nn.Sequential(layer1, *intermediate_layers, self.relu, layer2)


class NerfNaive(pl.LightningModule):
    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray: int,
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
        self.samples_per_ray = samples_per_ray
        
        # Hyper parameters for the optimizer 
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay
        
        # The model 
        self.model = NerfModel(
            n_hidden=10,
            hidden_dim=256
        )



    def _sample_t(self, batch_size: int) -> th.Tensor:
        """
        Sample t for sampling along the rays. Divides interval into equally sized bins, and samples a random point in each bin.

        Parameters:
        -----------
            batch_size: int - the amount of rays in a batch

        Returns:
        -----------
            t: Tensor of shape (batch_size, samples_per_ray)
        """
        # n = samples_per_ray
        # Calculate the interval size (divide far - near into n bins)
        interval_size = (self.far_sphere_normalized - self.near_sphere_normalized) / self.samples_per_ray
        
        # Sample n points with equal distance, starting at the near sphere and ending at one interval size before the far sphere
        t = th.linspace(
            self.near_sphere_normalized, 
            self.far_sphere_normalized - interval_size, 
            self.samples_per_ray, 
            device=self.device
        ).unsqueeze(0).repeat(batch_size, 1)
        
        # Perturb sample positions to be anywhere in each interval
        t += th.rand_like(t, device=self.device) * interval_size

        return t
    

    def _compute_positions(self, origins: th.Tensor, directions: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Compute the positions, distances and directions for given rays and t values.

        Parameters:
        -----------
            origins: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            directions: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.


        Returns:
        --------
            positions: Tensor of shape (batch_size, samples_per_ray, 3) - the positions for the given t values
            distances: Tensor of shape (batch_size, samples_per_ray) - the distances between the t values - used for the weights
                        Appended is the distance between the last position and the far sphere.
            directions: Tensor of shape (batch_size, samples_per_ray, 3) - the directions for the given t values
        """
        # Calculates the position with the equation p = o + t*d 
        # The unsqueeze is because origins and directions are spacial vectors while t is the sample times
        positions = origins.unsqueeze(1) + t.unsqueeze(2) * directions.unsqueeze(1)
        
        # Get distances in t values
        distances = th.hstack((
            t[:,1:] - t[:,:-1],
            self.far_sphere_normalized - t[:,-1:]
        ))

        # Format the directions to match the found positions
        directions = directions.unsqueeze(1).repeat(1, positions.shape[1], 1)
        
        return positions, directions, distances



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
                            t: th.Tensor,
                            ray_origs: th.Tensor,
                            ray_dirs: th.Tensor,
                            batch_size: int,
                            samples_per_ray: int) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        
        """
        A helper function to compute the color for given model, t values, ray origins and ray directions.
        Is just a wrapper around _compute_positions, model_base.forward and _render_rays.

        Parameters:
        -----------
            model: The model to use for computing the color 
            t: Tensor of shape (batch_size, samples_per_ray) - the t values to compute the positions for.
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
        sample_pos, sample_dir, sample_dist  = self._compute_positions(ray_origs, ray_dirs, t)

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



    def forward(self, ray_origs: th.Tensor, ray_dirs: th.Tensor) -> th.Tensor:
        """
        Forward pass of the model.
        Given the ray origins and directions, compute the rgb values for the given rays.

        Parameters:
        -----------
            ray_origs: Tensor of shape (batch_size, 3) - camera position, i.e. origin in camera coordinates
            ray_dirs: Tensor of shape (batch_size, 3) - direction vectors (unit vectors) of the viewing directions

        Returns:
        --------
            rgb: Tensor of shape (batch_size, 3) - the rgb values for the given rays using fine model (the actual prediction)
        """

        # Amount of pixels to render
        batch_size = ray_origs.shape[0]
        
        # sample t
        t_coarse = self._sample_t(batch_size)

        #compute rgb
        rgb, _, _ = self._compute_color(self.model,
                                                    t_coarse,
                                                    ray_origs,
                                                    ray_dirs,
                                                    batch_size,
                                                    self.samples_per_ray)
        
        # Return colors for the given pixel coordinates (batch_size, 3)
        return rgb


    ############ pytorch lightning functions ############

    def _step_helpher(self, batch: DatasetOutput, batch_idx: int, stage: str):
        """
        general function for training and validation step
        """
        ray_origs, ray_dirs, ray_colors = batch
        
        ray_colors_pred = self(ray_origs, ray_dirs)

        loss = nn.functional.mse_loss(ray_colors_pred, ray_colors)
        self.log(f"{stage}_loss", loss*2)

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


