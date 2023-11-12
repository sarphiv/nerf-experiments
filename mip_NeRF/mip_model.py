import torch as th
import pytorch_lightning as pl
from torch import nn

from typing import cast, Optional, Tuple

from model_interpolation_architecture import NerfModel, FourierFeatures
from model_interpolation import NerfInterpolation



class IntegratedFourierFeatures(FourierFeatures):

    def __init__(self, 
        levels: int,
        scale: float = 2*th.pi,
        distribute_variance: Optional[bool] = False,
    ):
        super().__init__(levels, scale)
        self.distribute_variance = distribute_variance
        self.output_dim += 3

    def forward(self,
                pos: th.Tensor,
                dir: th.Tensor,
                t_start: th.Tensor,
                t_end: th.Tensor,
                pixel_width: float,
                distribute_variance: Optional[bool] = None,
                ) -> th.Tensor:
        
        batch_size, space_dim = pos.shape
        distribute_variance = distribute_variance or self.distribute_variance
        assert space_dim == 3, "Only 3D supported"

        # Mip nerf helper constants
        t_mu = (t_start + t_end) / 2 # below eq 7
        t_delta = (t_end - t_start) / 2 # below eq 7

        # compute updated position (mu)
        mu_diff = 2*t_mu*t_delta**2/(3*t_mu**2+t_delta**2) # eq 8
        pos_mu = pos + mu_diff*dir # eq 8

        # compute diag_sigma_gamma
        # helper constants
        # Lauge: I think this is wrong:
        # frustum_radius = pixel_width * (t_start + mu_diff)
        # r_dot = 2/12**0.5*frustum_radius # above eq 5
        # Lauge: I think this is right:
        r_dot = pixel_width * 2 / (12**0.5) # above eq 5

        sigma_t_sq = t_delta**2/3 - (4*t_delta**4*(12*t_mu**2 - t_delta**2))/(15*(3*t_mu**2 + t_delta**2)**2) # eq 7
        sigma_r_sq = r_dot**2*(t_mu**2/4 + 5*t_delta**2/12-4*t_delta**4/(15*(3*t_mu**2+t_delta**2))) # eq 7

        scale = 4**th.arange(self.levels, device=pos.device).repeat(space_dim) # [4,16,4,16,4,16]

        if distribute_variance:
            Sigma = (sigma_t_sq + sigma_r_sq * 2)/space_dim*scale
            weight = th.exp(-Sigma/2)

        else:
            # calculate that diagonal
            diag_Sigma = sigma_t_sq*dir**2 + sigma_r_sq*(1-dir**2/th.sum(dir**2, dim=1, keepdim=True)) # eq 16

            # repeat and multiply by 4**i
            tmp = diag_Sigma.repeat_interleave(self.levels, dim=1) # with levels = 2 we get [x,x,y,y,z,z] ... times batch size
            diag_Sigma_gamma = tmp*scale # = [4x, 16x, 4y, ... ] times batch size.

            # compute positional encoding
            weight = th.exp(-diag_Sigma_gamma/2) # eq 14

        # call self.super.forward
        pe = super(IntegratedFourierFeatures, self).forward(pos_mu)

        ipe = pe*weight.repeat(1,2)

        return th.cat((pos, ipe), dim=1)
        


class MipNerfModel(NerfModel):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
        fourier: tuple[bool, int, int],
        n_segments: int,
        distribute_variance: Optional[bool] = False,
    ):
        # super().__init__(
        #     n_hidden,
        #     hidden_dim,
        #     fourier,
        #     delayed_direction=True,
        #     delayed_density=False,
        #     n_segments=n_segments,
        #         )

        # if self.fourier:
        #     fourier_levels_pos = self.position_encoder.levels
        #     self.position_encoder = IntegratedFourierFeatures(fourier_levels_pos, 2*th.pi, distribute_variance)

        super(NerfModel, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.delayed_direction = True
        self.delayed_density = False
        self.n_segments = n_segments
        self.fourier, fourier_levels_pos, fourier_levels_dir = fourier
        
        # If there is a positional encoding define it here
        if self.fourier:
            self.position_encoder = IntegratedFourierFeatures(fourier_levels_pos, 2*th.pi, distribute_variance)
            self.direction_encoder = FourierFeatures(fourier_levels_dir, 1.0)

            # Dimensionality of the input to the network 
            positional_dim = self.position_encoder.output_dim
            directional_dim = self.direction_encoder.output_dim
        else: 
            positional_dim = 3
            directional_dim = 3 
        
        # Create list of model segments
        self.model_segments = nn.ModuleList()
        for i in range(self.n_segments):
            # Create the model segment
            input_size = positional_dim + (not self.delayed_direction)*directional_dim + (i>0)*self.hidden_dim
            model_segment = self.contruct_model_density(input_size,
                                                        self.hidden_dim,
                                                        self.hidden_dim + (not self.delayed_density)*(i == self.n_segments-1))
            
            # Add the model segment to the list of model segments
            self.model_segments.append(model_segment)
        
        # Creates the final layer of the model that outputs the color
        self.model_color = nn.Sequential(
            nn.Linear(self.hidden_dim + self.delayed_direction*directional_dim, self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim//2, 3 + self.delayed_density)
        )
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus(threshold=8)
        self.sigmoid = nn.Sigmoid()


    def forward(self, pos: th.Tensor, dir: th.Tensor, t_start: th.Tensor, t_end: th.Tensor, pixel_width: th.Tensor, cam_idx: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    # def forward(self, pos, dir, t_start, t_end, pixel_width):
        if self.fourier:
            pos = self.position_encoder(pos, dir, t_start, t_end, pixel_width)
            dir = self.direction_encoder(dir)
        density, rgb = self._network_forward(pos, dir)
        return density, rgb
    
class MipNerf(NerfInterpolation):#, pl.LightningModule):

    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        samples_per_ray: int,
        n_hidden: int,
        proposal: tuple[bool, int],
        fourier: tuple[bool, int, int],
        n_segments: int,
        learning_rate: float = 1e-4,
        learning_rate_decay: float = 0.5,
        weight_decay: float = 0.0,
        distribute_variance: Optional[bool] = False,
    ): 
        
        # super(pl.LightningModule, self).__init__()
        super(NerfInterpolation, self).__init__()
        self.save_hyperparameters()

        # Hyper parameters for the optimizer 
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay

        # The near and far sphere distances 
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized
        

        # If there is a proposal network separate the samples into coarse and fine sampling
        self.proposal = proposal[0]
        if self.proposal:
            self.samples_per_ray_coarse = proposal[1]
            self.samples_per_ray_fine = samples_per_ray - proposal[1]

        else: 
            self.samples_per_ray_coarse = samples_per_ray
            self.samples_per_ray_fine = 0
        
        # build the model(s)
        self.model_fine = MipNerfModel(
            n_hidden,
            hidden_dim=256,
            fourier=fourier,
            n_segments=n_segments,
            distribute_variance=distribute_variance,
        )

        self.model_coarse = self.model_fine
