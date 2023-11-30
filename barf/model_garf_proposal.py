from typing import Iterator

import torch as th
import torch.nn as nn

from gaussian import GaussAct
from model_interpolation_architecture import NerfBaseModel


class ProposalNetwork(NerfBaseModel):
    def __init__(self, 
        gaussian_init_min: float,
        gaussian_init_max: float,
        learning_rate_start: float,
        learning_rate_stop: float,
        learning_rate_decay_end: float, # In steps
        gaussian_learning_rate_factor: float,
        weight_decay: float
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

        self._add_param_group(
            self.parameters_linear(), 
            learning_rate_start, 
            learning_rate_stop, 
            learning_rate_decay_end,
            weight_decay=weight_decay
        )

        self._add_param_group(
            self.parameters_gaussian(),
            learning_rate_start * gaussian_learning_rate_factor,
            learning_rate_stop * gaussian_learning_rate_factor,
            learning_rate_decay_end,
            weight_decay=weight_decay
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
