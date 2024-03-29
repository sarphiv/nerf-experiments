from typing import Iterator, Literal, Optional

import torch as th
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = None
        self.space_dimensions = None
    
    def forward(self, x: th.Tensor, dir: th.Tensor|None=None, pixel_width: th.Tensor|None=None, t_start: th.Tensor|None=None, t_end: th.Tensor|None=None) -> th.Tensor:
        raise NotImplementedError()


class IdentityPositionalEncoding(PositionalEncoding):
    def __init__(self, space_dimensions: int = 3):
        super().__init__()
        self.output_dim = space_dimensions
        self.space_dimensions = space_dimensions

    def forward(self, x: th.Tensor, dir=None, pixel_width=None, t_start=None, t_end=None) -> th.Tensor:
        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"
        return x


class FourierFeatures(PositionalEncoding):
    def __init__(self, levels: int, scale: float = 2*th.pi, space_dimensions: int = 3): 
        """
        Positional encoding using Fourier features of "levels" periods. 
        
        """
        super().__init__()

        self.levels = levels
        self.scale = scale
        self.space_dimensions = space_dimensions
        self.output_dim = levels*2*space_dimensions


    def forward(self, x: th.Tensor, dir=None, pixel_width=None, t_start=None, t_end=None) -> th.Tensor:
        """
        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for i in [0, levels-1]

        returns:
            [cos(x), cos(2x), cos(4x) ... , cos(y), cos(2y), cos(4y) ... , cos(z), cos(2z), cos(4z) ...,
             sin(x), sin(2x), sin(4x) ... , sin(y), sin(2y), sin(4y) ... , sin(z), sin(2z), sin(4z) ...]
        """

        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"

        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(x.shape[1])
        args = x.repeat_interleave(self.levels, dim=1) * scale

        return th.hstack((th.cos(args), th.sin(args)))



class BarfPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 levels: int,
                 alpha_start: float,
                 alpha_increase_start_epoch: float,
                 alpha_increase_end_epoch: float,
                 include_identity: bool = True,
                 scale: float = 2*th.pi,
                 space_dimensions: int = 3):
        """
        Positional encoding using the mask from the barf paper. 
        """
        super().__init__()
        self.levels = levels
        self.alpha_start = alpha_start
        self.output_dim = (levels*2 + include_identity)*space_dimensions
        self.alpha_increase_start_epoch = alpha_increase_start_epoch
        self.alpha_increase_end_epoch = alpha_increase_end_epoch
        self.include_identity = include_identity
        self.scale = scale
        self.space_dimensions = space_dimensions
        self.register_buffer("alpha", th.tensor(float(alpha_start)))

    def update_alpha(self, epoch: float) -> None:
        """
        Updates the alpha value of the positional encoding. 
        """

        if epoch < self.alpha_increase_start_epoch:
            alpha = self.alpha_start

        elif self.alpha_increase_start_epoch <= epoch < self.alpha_increase_end_epoch:
            alpha = (
                self.alpha_start
                + (epoch - self.alpha_increase_start_epoch)
                * (self.levels - self.alpha_start)
                / (self.alpha_increase_end_epoch - self.alpha_increase_start_epoch)
            )

        else:
            alpha = float(self.levels)
        
        self.alpha = th.tensor(alpha, device=self.alpha.device)
    
    def compute_mask(self, alpha: th.Tensor) -> th.Tensor:
        # init zero mask
        mask = th.zeros((self.levels), device=alpha.device)

        # identify the turning point, k where 1 > alpha - k > 0 mask 
        idx_ramp = int(alpha)

        # set ones
        mask[:idx_ramp] = 1.

        # the turning point is a cosine interpolation
        if idx_ramp < self.levels:
            mask[idx_ramp] = (1 - th.cos((alpha - idx_ramp) * th.pi)) / 2

        # repeat for sin and for each channel
        mask = mask.repeat(self.space_dimensions)
    
        return mask.view(1, -1)
        
    def forward(self, x: th.Tensor, dir=None, pixel_width=None, t_start=None, t_end=None) -> th.Tensor:
        """

        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for i in [0, levels-1]

        returns:
            [cos(x), cos(2x), cos(4x) ... , cos(y), cos(2y), cos(4y) ... , cos(z), cos(2z), cos(4z) ...,
             sin(x), sin(2x), sin(4x) ... , sin(y), sin(2y), sin(4y) ... , sin(z), sin(2z), sin(4z) ...]

            if include_identity:
                [x,y,z] is prepended to the output
        """

        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"

        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(self.space_dimensions)
        args = x.repeat_interleave(self.levels, dim=1) * scale

        mask = self.compute_mask(self.alpha)

        if self.include_identity:
            return th.hstack((x, mask*th.cos(args), mask*th.sin(args)))
        else:
            return th.hstack((mask*th.cos(args), mask*th.sin(args)))

# TODO maybe fix such that it takes scale into account
class IntegratedFourierFeatures(PositionalEncoding):

    def __init__(self, 
        levels: int,
        scale: float = 2*th.pi,
        include_identity = True,
        distribute_variance: Optional[bool] = False,
    ):
        super().__init__()
        self.levels = levels
        self.space_dimensions = 3
        self.scale = scale
        self.include_identity = include_identity
        self.output_dim = (levels*2 + include_identity)*self.space_dimensions
        self.distribute_variance = distribute_variance
        self.pixel_width_sigma =  None



    def forward(self,
                pos: th.Tensor,
                dir: th.Tensor,
                pixel_width: th.Tensor,
                t_start: th.Tensor,
                t_end: th.Tensor,
                include_identity: bool|None = None,
                diagnose: bool = False
                ) -> th.Tensor:
        
        include_identity = include_identity if include_identity is not None else self.include_identity 

        batch_size, space_dim = pos.shape
        if not space_dim == 3: raise ValueError(f"Only 3D supported - was {space_dim}D")

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

        add_sigma = (self.pixel_width_sigma*pixel_width*t_mu)**2 if self.pixel_width_sigma > 0.25 else 0.

        sigma_t_sq = sigma_t_sq + add_sigma
        sigma_r_sq = sigma_r_sq + add_sigma


        scale = 4**th.arange(self.levels, device=pos.device).repeat(space_dim) # [4,16,4,16,4,16]


        if self.distribute_variance:
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
        pe = FourierFeatures.forward(self, pos_mu)

        ipe = pe*weight.repeat(1,2)

        if include_identity:
            ipe = th.cat((pos_mu, ipe), dim=1)

        if diagnose:
            return ipe, weight, sigma_t_sq, sigma_r_sq
        else:
            return ipe

class IntegratedBarfFourierFeatures(BarfPositionalEncoding):

    def __init__(self,
                 levels: int,
                 alpha_start: float,
                 alpha_increase_start_epoch: float,
                 alpha_increase_end_epoch: float,
                 include_identity: bool = True,
                 scale: float = 2*th.pi,
                 distribute_variance = True):

        BarfPositionalEncoding.__init__(self,
            levels=levels,
            alpha_start=alpha_start,
            alpha_increase_start_epoch=alpha_increase_start_epoch,
            alpha_increase_end_epoch=alpha_increase_end_epoch,
            include_identity=include_identity,
            scale=scale,
            space_dimensions=3
        )

        self.distribute_variance = distribute_variance


    def forward(self,
                pos: th.Tensor,
                dir: th.Tensor,
                pixel_width: th.Tensor,
                t_start: th.Tensor,
                t_end: th.Tensor,
                ) -> th.Tensor:
        
        mask = BarfPositionalEncoding.compute_mask(self, self.alpha)
        ipe = IntegratedFourierFeatures.forward(self, pos, dir, pixel_width, t_start, t_end, self.include_identity)
        if self.include_identity:
            size = self.levels*self.space_dimensions
            pos, cos, sin = th.split(ipe, (self.space_dimensions, size, size), dim=1)
            return th.hstack((pos, mask*cos, mask*sin))
        else:
            cos, sin = th.split(ipe, (size, size), dim=1)
            return th.hstack((mask*cos, mask*sin))
     


if __name__ == "__main__":
    pos = th.randn(1000, 3)*3
    dir = th.randn(1000, 3)
    dir = dir/th.linalg.vector_norm(dir, dim=1, keepdim=True)
    t_start = th.rand(1000, 1)*4
    t_end = t_start + 1e-3# + th.rand_like(t_start)

    alpha = 0.1
    include_identity = True
    scale = 1.

    # ipe = IntegratedFourierFeatures(10, scale, include_identity, False)
    pe_barf = BarfPositionalEncoding(10, alpha, 12,13, include_identity, scale, 3)
    ipe_barf = IntegratedBarfFourierFeatures(10, alpha, 12, 13, include_identity, scale, False)

    out_new = pe_barf.forward(pos.clone(), dir.clone(), 0.001, t_start.clone(), t_end.clone())
    out_old = ipe_barf.forward(pos.clone(), dir.clone(), 0.001, t_start.clone(), t_end.clone())

    diff_max = th.max(out_new - out_old).detach().item()
    diff_min = th.min(out_new - out_old).detach().item()

    d = th.max((out_new - out_old).abs()).detach().item()
    
    print(d)

    diff = out_new - out_old
    x = (t_end - t_start).squeeze()

    import matplotlib.pyplot as plt

    plt.plot(x, diff[:,22], ".")
    plt.savefig("teeeeeeeeeeeeeeeeeeeeeest.png")
