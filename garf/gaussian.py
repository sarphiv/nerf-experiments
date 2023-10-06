import torch as th
import torch.nn as nn


class GaussActivation(th.autograd.Function):

    @staticmethod
    def forward(ctx, x, variance):
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
    def __init__(self, size: int):
        # Initialize parameters
        super().__init__()
        self.act_var = nn.Sigmoid()

        self.variance = nn.Parameter(th.rand(size) + 2)
        # NOTE: Need the softplus to ensure variance is positive
        # self.act_var = lambda x: th.exp(x)
        # self.act_var = lambda x: x
        self.act_var = lambda x: x**2 + 1.e-5
        self.func = GaussActivation.apply
        

    def forward(self, x: th.Tensor):
        return self.func(x, self.act_var(self.variance))
