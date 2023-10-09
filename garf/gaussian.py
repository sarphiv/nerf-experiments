from typing import cast

import torch as th
import torch.nn as nn
from torch.autograd.function import FunctionCtx


class GaussActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: th.Tensor, variance: th.Tensor):
        # Save parameters
        ctx.save_for_backward(x, variance)

        # Compute output
        return th.exp(-x**2 / (2*variance))

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: th.Tensor):
        # Retrieve parameters
        x, variance = cast(tuple[th.Tensor, th.Tensor], ctx.saved_tensors) # type: ignore

        # Compute gradients
        x2 = x**2
        exp = th.exp(-x2 / (2*variance))
        grad_exp = grad_output * exp

        grad_input_x = -grad_exp * x / variance
        grad_input_variance = grad_exp * x2 / (2*variance**2)

        return grad_input_x, grad_input_variance


class GaussAct(nn.Module):
    def __init__(
        self,
        features_in: int,
        standard_deviation_init_min: float = 0.,
        standard_deviation_init_max: float = 1.
    ):
        """
        Gaussian activation function with learnable variance.

        Parameters:
        ----------
        features_in: int
            Number of input features
        standard_deviation_init_min: float
            Minimum initial value for the standard deviation initiated via uniform distribution
        standard_deviation_init_max: float
            Maximum initial value for the standard deviation initiated via uniform distribution
        """
        # Initialize parameters
        super().__init__()

        #NOTE: Negative standard_deviation is allowed to be negative as we only ever use var = std**2
        self.standard_deviation = nn.Parameter(
            th.rand(features_in) * (standard_deviation_init_max - standard_deviation_init_min) + standard_deviation_init_min
        )


    def forward(self, x: th.Tensor):
        return GaussActivation.apply(x, (self.standard_deviation**2).clamp_min(1e-6))
