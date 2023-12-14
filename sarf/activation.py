from typing import cast

import torch as th
import torch.nn as nn
from torch.autograd.function import FunctionCtx


class SarfActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx: FunctionCtx, x: th.Tensor, f: th.Tensor):
        # Avoid division by zero
        # NOTE: Approaches f for x = 0, so small numbers should be fine
        x = th.sign(x) * (th.abs(x) + th.finfo(x.dtype).tiny)

        # Save parameters
        ctx.save_for_backward(x, f)

        # Compute output
        return th.sin(f*x) / x


    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: th.Tensor):
        # Retrieve parameters
        # NOTE: Forward pass ensures x != 0
        x, f = cast(tuple[th.Tensor, th.Tensor], ctx.saved_tensors) # type: ignore

        # Return gradients
        return (
            grad_output * (f*th.cos(f*x)*x - th.sin(f*x)) / x**2,
            grad_output * th.cos(f*x)
        )


class SarfAct(nn.Module):
    def __init__(
            self,
            features_in: int,
            frequency_init_min: float = 0.,
            frequency_init_max: float = 1.
        ):
            """
            Initialize the SARF activation function.

            Args:
                features_in (int): Number of input features.
                frequency_init_min (float, optional): Minimum value for initializing the frequency. Defaults to 0.
                frequency_init_max (float, optional): Maximum value for initializing the frequency. Defaults to 1.
            """
            # Initialize parameters
            super().__init__()

            self.frequency = nn.Parameter(
                th.rand(features_in) * (frequency_init_max - frequency_init_min) + frequency_init_min
            )


    def forward(self, x: th.Tensor):
        return SarfActivation.apply(x, self.frequency)
