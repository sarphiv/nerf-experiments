from typing import Iterator

import torch as th
import torch.nn as nn

from gabor import GaborAct


class RadianceNetwork(nn.Module):
    def __init__(self, 
        gaussian_init_min: float,
        gaussian_init_max: float,
    ):
        super().__init__()
        self.gaussian_init_min = gaussian_init_min
        self.gaussian_init_max = gaussian_init_max
        
        self._parameters_linear: list[nn.Parameter] = []
        self._parameters_gabor: list[nn.Parameter] = []


        # Creates the first module of the network 
        self.model_density_1 = nn.Sequential(
            self._create_linear(3, 1024),
            self._create_gabor(1024),
            self._create_linear(1024, 256),
            self._create_gabor(256),
            self._create_linear(256, 128),
            self._create_gabor(128),
            self._create_linear(128, 128),
            self._create_gabor(128),
        )

        # Creates the second module of the network (skip connection of input to the network again)
        self.model_density_2 = nn.Sequential(
            self._create_linear(128 + 3, 512),
            self._create_gabor(512),
            self._create_linear(512, 256),
            self._create_gabor(256),
            self._create_linear(256, 128),
            self._create_gabor(128),
            self._create_linear(128, 128 + 1)
        )

        # Creates activation function for density
        self.softplus = nn.Softplus(threshold=8)

        # Creates the final module of the model that outputs the color
        self.model_color = nn.Sequential(
            self._create_linear(128 + 3, 256),
            self._create_gabor(256),
            self._create_linear(256, 3),
            nn.Sigmoid()
        )


    def _create_linear(self, features_in: int, features_out: int) -> nn.Linear:
        linear = nn.Linear(features_in, features_out)
        self._parameters_linear.append(linear.weight) # type: ignore
        self._parameters_linear.append(linear.bias)

        return linear

    def _create_gabor(self, features_in) -> GaborAct:
        act = GaborAct(features_in, self.gaussian_init_min, self.gaussian_init_max)
        self._parameters_gabor.append(act.inv_standard_deviation)
        self._parameters_gabor.append(act.spread)

        return act


    def parameters_linear(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_linear)

    def parameters_gabor(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_gabor)


    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        z1 = self.model_density_1(pos)
        z2 = self.model_density_2(th.cat((z1, pos), dim=1))

        # Extract density
        density = self.softplus(z2[:, 128] - 1)

        # Extract the color estimate
        rgb = self.model_color(th.cat((z1[:, :128] + z2[:, :128], dir), dim=1))


        # Return estimates
        return rgb, density
