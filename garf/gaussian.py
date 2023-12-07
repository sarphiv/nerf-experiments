from typing import cast

import torch as th
import torch.nn as nn
from torch.autograd.function import FunctionCtx


# class GaussActivation(th.autograd.Function):

#     @staticmethod
#     def forward(ctx: FunctionCtx, x: th.Tensor, inv_variance: th.Tensor):
#         # Save parameters
#         ctx.save_for_backward(x, inv_variance)

#         # Compute output
#         return th.exp(-x**2 * inv_variance)

#     @staticmethod
#     def backward(ctx: FunctionCtx, grad_output: th.Tensor):
#         # Retrieve parameters
#         x, inv_variance = cast(tuple[th.Tensor, th.Tensor], ctx.saved_tensors) # type: ignore

#         # Compute gradients
#         x2 = x**2
#         exp = th.exp(-x2 * inv_variance)
#         grad_exp = grad_output * exp

#         grad_input_x = -grad_exp * 2 * x * inv_variance
#         grad_input_inv_variance = -grad_exp * x2

#         return grad_input_x, grad_input_inv_variance


class GaussAct(nn.Module):
    def __init__(
        self,
        features_in: int,
        inv_standard_deviation_init_min: float = 0.,
        inv_standard_deviation_init_max: float = 1.
    ):
        """
        Gaussian activation function with learnable variance.

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


    def forward(self, x: th.Tensor):
        return th.exp(-x**2 * 0.5*5**2)
        # return GaussActivation.apply(x, self.inv_standard_deviation**2 + 1e-6)
