import torch as th
import torch.nn as nn



class GaussAct(nn.Module):
    def __init__(self, variance: float):
        # Initialize parameters
        if isinstance(variance, int):
            variance = float(variance)

        if not isinstance(variance, float):
            raise TypeError("Scale must be either a float or a tensor.")
        
        if variance <= 0:
            raise ValueError("Scale must be positive.")


        super().__init__()
        self.variance = variance


    def forward(self, x: th.Tensor):
        return th.exp(x**2 / (-2*self.variance))

