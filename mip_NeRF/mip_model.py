import torch as th
import pytorch_lightning as pl

from typing import cast

from model_interpolation_architecture import NerfModel, FourierFeatures
from model_interpolation import NerfInterpolation



class IntegratedFourierFeatures(FourierFeatures):
    def forward(self, pos: th.Tensor, dir: th.Tensor, t_start: th.Tensor, t_end: th.Tensor, pixel_width: float) -> th.Tensor:
        
        # Mip nerf helper constants
        t_mu = (t_start + t_end) / 2 # below eq 7
        t_delta = (t_end - t_start) / 2 # below eq 7

        # compute updated position (mu)
        mu_diff = 2*t_mu*t_delta**2/(3*t_mu**2+t_delta) # eq 8
        pos_mu = pos + mu_diff*dir # eq 8

        # compute diag_sigma_gamma
        # helper constants
        frustum_radius = pixel_width * (t_start + mu_diff)
        frustum_radius = frustum_radius.to()
        r_dot = 2/12**0.5*frustum_radius # above eq 5

        sigma_t_sq = t_delta**2/3 - (4*t_delta**4*(12*t_mu**2 - t_delta**2))/(15*(3*t_mu**2 + t_delta**2)**2) # eq 7
        sigma_r_sq = r_dot**2*(t_mu**2/4 + 5*t_delta**2/12-4*t_delta**4/(15*(3*t_mu**2+t_delta**2))) # eq 7

        # calculate that diagonal
        diag_sigma = sigma_t_sq*dir**2 + sigma_r_sq*(1-dir**2/th.sum(dir**2, dim=1, keepdim=True)) # eq 16
        diag = diag_sigma.repeat(1, self.levels*2)*4**th.arange(self.levels, device=diag_sigma.device).repeat(pos.shape[0], 2).repeat_interleave(3, 1) # eq 15

        # compute positional encoding
        weight = th.exp(-diag/2) # eq 14
        # call selv.super.forward
        pos_enc = super(IntegratedFourierFeatures, self).forward(pos_mu)*weight

        return pos_enc
        


class MipNerfModel(NerfModel):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
        fourier: tuple[bool, int, int],
        n_segments: int
    ):
        super().__init__(
            n_hidden,
            hidden_dim,
            fourier,
            delayed_direction=True,
            delayed_density=False,
            n_segments=n_segments,
                )
        
        if self.fourier:
            fourier_levels_pos = self.position_encoder.levels
            self.position_encoder = IntegratedFourierFeatures(fourier_levels_pos, 2*th.pi)

    def forward(self, pos, dir, t_start, t_end, pixel_width):
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
        weight_decay: float = 0.0
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
        
        self.model = MipNerfModel(
            n_hidden,
            hidden_dim=256,
            fourier=fourier,
            n_segments=n_segments
        )

        self.model_coarse = self.model
        self.model_fine = self.model
