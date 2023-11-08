from typing import Iterator

import torch as th
import torch.nn as nn

from gaussian import GaussAct


class ProposalNetwork(nn.Module):
    def __init__(self, 
        gaussian_init_min: float,
        gaussian_init_max: float,
    ):
        super().__init__()
        self.gaussian_init_min = gaussian_init_min
        self.gaussian_init_max = gaussian_init_max
        
        self._parameters_linear: list[nn.Parameter] = []
        self._parameters_gaussian: list[nn.Parameter] = []

        self.model = nn.Sequential(
            self._create_linear(3, 512),
            self._create_gaussian(512),
            self._create_linear(512, 256),
            self._create_gaussian(256),
            self._create_linear(256, 128),
            self._create_gaussian(128),
            self._create_linear(128, 1),
            nn.Softplus(threshold=8)
        )


    def _create_linear(self, features_in: int, features_out: int) -> nn.Linear:
        linear = nn.Linear(features_in, features_out)
        self._parameters_linear.append(linear.weight) # type: ignore
        self._parameters_linear.append(linear.bias)

        return linear

    def _create_gaussian(self, features_in) -> GaussAct:
        act = GaussAct(features_in, self.gaussian_init_min, self.gaussian_init_max)
        self._parameters_gaussian.append(act.inv_standard_deviation)

        return act


    def parameters_linear(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_linear)

    def parameters_gaussian(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_gaussian)


    def forward(self, pos: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.model(pos)
