import torch as th
import torch.nn as nn
from typing import cast, Callable


class GaussActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx, x, variance_inv) -> th.Tensor:
        # Save parameters
        ctx.save_for_backward(x, variance_inv)

        # Compute output
        return th.exp(-x**2*2*variance_inv)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve parameters
        x, variance_inv = ctx.saved_tensors

        # Compute gradients
        x2 = x**2
        exp = th.exp(-x2*2*variance_inv)
        grad_exp = grad_output * exp

        grad_input_x = -grad_exp * x * variance_inv * 4
        grad_input_variance_inv = -grad_exp * x2 * 2

        return grad_input_x, grad_input_variance_inv


class GaussAct(nn.Module):
    def __init__(self, size: int, initial_values: Callable):
        # Initialize parameters
        super().__init__()
        #NOTE: negative standard_deviation is allowed to be negative as we only ever use var = std**2
        self.act_param = lambda x: x**2
        self.func = GaussActivation.apply
        self.param = nn.Parameter(initial_values(size))
        self.loss: th.Tensor = th.tensor(0)
    #     self.register_full_backward_hook(hook=GaussAct.hook)
    
    # @staticmethod
    # def hook(module, grad_input, grad_output):
    #     module.loss = th.tensor(0)
    #     return grad_input, grad_output

    # def forward(self, x: th.Tensor, scale=0.) -> th.Tensor:
    #     if self.standard_deviation is None:
    #         self.setup(x)
    #     var = self.variance
    #     self.loss = self.loss + th.mean((x - th.sqrt(var))**2)

    #     return cast(th.Tensor, self.func(x, var))

    def forward(self, x: th.Tensor, scale=0.) -> th.Tensor:
        var_inv = self.act_param(self.param)
        freq_represent = th.sqrt(var_inv)/6
        if scale == 0:
            weight = 1.
        else:
            freq_scale = 1/scale
            # weight = (1 - th.cos(th.pi*(freq_represent-freq_scale*2)/(freq_scale)))/2
            # weight[freq_represent < freq_scale] = 1.
            # weight[freq_represent > freq_scale*2] = 0.
            weight = th.sigmoid(3*th.pi - 2*th.pi*freq_represent/freq_scale).detach()

        return cast(th.Tensor, self.func(x, var_inv))*weight
    
if __name__ == "__main__":
        freq_represent = th.linspace(0, 1, 1000)
        freq_scale = 0.5
        weight1 = (1 - th.cos(th.pi*(freq_represent-freq_scale*2)/(freq_scale)))/2
        weight1[freq_represent < freq_scale] = 1.
        weight1[freq_represent > freq_scale*2] = 0.
        weight2 = th.sigmoid(3*th.pi - 2*th.pi*freq_represent/freq_scale)
        import matplotlib.pyplot as plt
        plt.plot(freq_represent, weight1)
        plt.plot(freq_represent, weight2)
        plt.show()