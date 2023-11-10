from typing import cast

import torch as th
import torch.nn as nn
from torch.autograd.function import FunctionCtx


class GaborActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: th.Tensor, inv_variance: th.Tensor, spread: th.Tensor):
        # Save parameters
        ctx.save_for_backward(x, inv_variance, spread)

        # Compute output
        return th.exp(-inv_variance * x**2) * th.cos(spread * x)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: th.Tensor):
        # Retrieve parameters
        x, v, s = cast(tuple[th.Tensor, th.Tensor], ctx.saved_tensors) # type: ignore

        # Compute gradients
        x2 = x**2
        go_mevx2 = -th.exp(-v*x2) * grad_output
        sinsx = th.sin(s*x)
        cossx = th.cos(s*x)
        
        grad_input_x = go_mevx2 * (2*cossx*v*x + s*sinsx)
        grad_input_v = go_mevx2 * x2 * cossx
        grad_input_s = go_mevx2 * x * sinsx

        return grad_input_x, grad_input_v, grad_input_s


class GaborAct(nn.Module):
    def __init__(
        self,
        features_in: int,
        inv_standard_deviation_init_min: float = 0.,
        inv_standard_deviation_init_max: float = 1.
    ):
        """
        Gabor activation function with learnable variance and spread.

        Parameters:
        ----------
        features_in: int
            Number of input features
        inv_standard_deviation_init_min: float
            minuimum initial value for the inverse standard deviation initiated via uniform distribution
        inv_standard_deviation_init_max: float
            Maximum initial value for the inverse standard deviation initiated via uniform distribution
        """
        # Initialize parameters
        super().__init__()

        #NOTE: Negative standard_deviation is allowed to be negative as we only ever use var = std**2
        self.inv_standard_deviation = nn.Parameter(
            th.rand(features_in) * (inv_standard_deviation_init_max - inv_standard_deviation_init_min) + inv_standard_deviation_init_min
        )
        self.spread = nn.Parameter(
            th.rand(features_in) * 2 * th.pi
        )


    def forward(self, x: th.Tensor):
        return GaborActivation.apply(x, self.inv_standard_deviation**2 + 1e-6, self.spread)
