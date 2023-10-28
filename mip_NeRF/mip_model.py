import torch as th

from typing import cast

from model_interpolation_architecture import NerfModel, FourierFeatures


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
        r_dot = 2/12**0.5*frustum_radius # above eq 5

        sigma_t_sq = t_delta**2/3 - (4*t_delta**4*(12*t_mu**2 - t_delta**2))/(15*(3*t_mu**2 + t_delta**2)**2) # eq 7
        sigma_r_sq = r_dot**2*(t_mu**2/4 + 5*t_delta**2/12-4*t_delta**4/(15*(3*t_mu**2+t_delta**2))) # eq 7

        # calculate that diagonal
        diag_sigma = sigma_t_sq*dir**2 + sigma_r_sq*(1-dir**2/th.sum(dir**2, dim=1, keepdim=True)) # eq 16
        diag = diag_sigma.repeat(1, self.levels*2)*4**th.arange(self.levels).repeat(pos.shape[0], 2).repeat_interleave(3, 1) # eq 15

        # compute positional encoding
        weight = th.exp(-diag/2) # eq 14
        # call selv.super.forward
        pos_enc = super().forward(pos_mu)*weight

        return pos_enc
        


class MipNerfModel(NerfModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.fourier:
            fourier_levels_pos = self.position_encoder.levels
            self.position_encoder = IntegratedFourierFeatures(fourier_levels_pos, 2*th.pi)

    def forward(self, pos, dir, t_start, t_end, pixel_width):
        if self.fourier:
            pos = self.position_encoder(pos, dir, t_start, t_end, pixel_width)
            dir = self.direction_encoder(dir)
        density, rgb = self._network_forward(pos, dir)
        return density, rgb
    
        
