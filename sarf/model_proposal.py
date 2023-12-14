from typing import Iterator

import torch as th
import torch.nn as nn

from activation import SarfAct


class ProposalNetwork(nn.Module):
    def __init__(
        self, 
        frequency_init_min: float,
        frequency_init_max: float,
    ):
        super().__init__()
        self.frequency_init_min = frequency_init_min
        self.frequency_init_max = frequency_init_max
        
        self._parameters_linear: list[nn.Parameter] = []
        self._parameters_activation: list[nn.Parameter] = []

        self.model = nn.Sequential(
            self._create_linear(3, 512),
            self._create_activation(512),
            self._create_linear(512, 256),
            self._create_activation(256),
            self._create_linear(256, 128),
            self._create_activation(128),
            self._create_linear(128, 1),
            nn.Softplus(threshold=8)
        )


    def _create_linear(self, features_in: int, features_out: int) -> nn.Linear:
        linear = nn.Linear(features_in, features_out)
        self._parameters_linear.append(linear.weight) # type: ignore
        self._parameters_linear.append(linear.bias)

        return linear

    def _create_activation(self, features_in) -> SarfAct:
        act = SarfAct(features_in, self.frequency_init_min, self.frequency_init_max)
        self._parameters_activation.append(act.frequency)

        return act


    def parameters_linear(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_linear)

    def parameters_activation(self) -> Iterator[nn.Parameter]:
        return iter(self._parameters_activation)


    def forward(self, pos: th.Tensor) -> th.Tensor:
        return self.model(pos)
