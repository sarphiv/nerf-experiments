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
        gauss_act = lambda x: GaussAct(x, gaussian_init_min, gaussian_init_max)

        self.model = nn.Sequential(
            nn.Linear(3, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 1),
            nn.Softplus(threshold=8)
        )


    def forward(self, pos: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.model(pos)
