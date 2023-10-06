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
    def __init__(self, variance_initial: float):
        # Initialize parameters
        if isinstance(variance_initial, int):
            variance_initial = float(variance_initial)

        if not isinstance(variance_initial, float):
            raise TypeError("Variance must be either a float or a tensor.")
        
        if variance_initial <= 0:
            raise ValueError("Variance must be positive.")

        super().__init__()
        self.variance = nn.Parameter(th.tensor(variance_initial))
        # NOTE: Need the softplus to ensure variance is positive
        self.softplus = nn.Softplus(10)
        self.func = GaussActivation.apply
        

    def forward(self, x: th.Tensor):
        return self.func(x, self.variance)
