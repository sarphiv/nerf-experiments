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
    def __init__(self,
                 size: int,
                 init_mean=0.,
                 init_std=1.,
                 initial_values=None,
                 parameter_activation=None,
                 initializer=None,
                 *init_args,
                 **init_kwargs):
    
        """
        Gaussian activation function with learnable variance.

        Parameters:
        ----------
        size: int
            Size of the parameter tensor (number of neurons in the layer)
        init_mean: float
            mean of the initial values. Ignored if initial_values or initializer is not None
        init_std: float
            standard deviation of the initial values. Ignored if initial_values or initializer is not None
        initial_values: th.Tensor
            initial values for the standard deviation. Must have shape (size, ) and overwrites init_mean and init_std
        parameter_activation: callable
            activation function applied to the parameter (self.standard_deviation) of the layer. If None,
            parameter_activation(x) = x**2 + 1e-6 is used.
        initializer: callable
            initializer function that takes (size, *init_args, **init_kwargs) as input and returns a tensor of shape (size, )
        init_args: tuple
            additional arguments for the initializer
        init_kwargs: dict
            additional keyword arguments for the initializer

        """
        # Initialize parameters
        super().__init__()
    
        if initial_values is not None:
            assert initial_values.shape == (size, ), f"initial_values must have shape (size, ), but has shape {initial_values.shape}"
        elif initializer is not None:
            initial_values = initializer(size, *init_args, **init_kwargs)
        else:
            initial_values = th.randn(size) * init_std + init_mean
        
        self.parameter_activation = parameter_activation or (lambda x: x**2 + 1e-4)
        
        #NOTE: negative standard_deviation is allowed to be negative as we only ever use var = std**2
        self.standard_deviation = nn.Parameter(initial_values)
        self.func = GaussActivation.apply
        

    def forward(self, x: th.Tensor):
        return self.func(x, self.parameter_activation(self.standard_deviation))
