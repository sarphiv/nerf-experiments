from math import sqrt

import torch as th
import torch.nn as nn



class LinearSine(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float | th.Tensor = 1.0, first_layer: bool = False):
        if isinstance(scale, int):
            scale = float(scale)

        if isinstance(scale, float):
            assert scale > 0, "Scale must be positive."
        elif isinstance(scale, th.Tensor):
            assert scale.shape[0] == in_features, "Scale must have the same dimensionality as input features."
            assert all(scale > 0), "Scale must be positive."
        else:
            raise TypeError("Scale must be either a float or a tensor.")

        super().__init__()

        # Scale is omega_0 in the paper
        self.register_buffer("scale", th.ones(in_features) * scale)
        self.first_layer = first_layer
        self.linear = nn.Linear(in_features, out_features)

        self.reset_parameters()


    @th.no_grad()
    def _init_weights(self, weight_width: th.Tensor):
        self.linear.weight.uniform_(-1, 1)
        self.linear.weight *= weight_width

    def reset_parameters(self) -> None:
        if self.first_layer:
            self._init_weights(1 / self.linear.in_features) # type: ignore
        else:
            self._init_weights(sqrt(6 / self.linear.in_features) / self.scale) # type: ignore
        


    def forward(self, x: th.Tensor):
        return th.sin(self.linear(self.scale * x))
