import torch as th
import torch.nn as nn
from typing import cast, Callable


class GaussActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx, x, variance) -> th.Tensor:
        # Save parameters
        ctx.save_for_backward(x, variance)

        # Compute output
        return th.exp(-x**2 / (2*variance))

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve parameters
        x, variance = ctx.saved_tensors

        # Compute gradients
        x2 = x**2
        exp = th.exp(-x2 / (2*variance))
        grad_exp = grad_output * exp

        grad_input_x = -grad_exp * x / variance
        grad_input_variance = grad_exp * x2 / (2*variance**2)

        return grad_input_x, grad_input_variance


class GaussAct(nn.Module):
    def __init__(self, initial_values: Callable):
        # Initialize parameters
        super().__init__()
        #NOTE: negative standard_deviation is allowed to be negative as we only ever use var = std**2
        self.act_var = lambda x: x**2 + 1e-6
        self.func = GaussActivation.apply
        self.standard_deviation = None
        self.initial_values = initial_values
    
    def setup(self, x: th.Tensor):
        self.standard_deviation = nn.Parameter(self.initial_values(x.shape))
    
    @property
    def variance(self) -> th.Tensor:
        return self.act_var(self.standard_deviation)

    def forward(self, x: th.Tensor, scale=0.) -> th.Tensor:
        var = self.variance
        freq_represent = 1/(6*th.sqrt(var))
        if scale == 0:
            weight = 1.
        else:
            freq_scale = 1/scale
            weight = (1 - th.cos(th.pi*(freq_represent-freq_scale*2)/(freq_scale)))/2
            weight[freq_represent < freq_scale] = 1.
            weight[freq_represent > freq_scale*2] = 0.

        return cast(th.Tensor, self.func(x, var))*weight
    
if __name__ == "__main__":
        freq_represent = th.linspace(0, 1, 1000)
        freq_scale = 0.5
        m1 = (freq_represent > freq_scale) & (freq_represent < freq_scale*2)
        m2 = (freq_represent > freq_scale/2) & (freq_represent < freq_scale)
        weight = th.zeros_like(freq_represent)
        weight[m1] = (1 - th.cos(th.pi*(freq_represent[m1]-freq_scale*2)/(freq_scale)))/2
        weight[m2] = (1 - th.cos(th.pi*(freq_represent[m2]-freq_scale/2)/(freq_scale/2)))/2
        import matplotlib.pyplot as plt
        plt.plot(freq_represent, weight)
        plt.show()