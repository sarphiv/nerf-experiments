import torch as th
import torch.nn as nn

from gaussian import GaussAct


class RadianceNetwork(nn.Module):
    def __init__(self, 
        gaussian_init_min: float,
        gaussian_init_max: float,
    ):
        """
        An instance of a NeRF model the architecture; 
        
           Hn( Hn(x), x )       -> density 
        H( Hn( Hn(x), x ), d )  -> color 

        H: a hidden fully connected layer with hidden_dim neurons,
        Hn: H applied n_hidden times, 
        x: is the position
        d: is the direction of the camera ray
        """
        super().__init__()
        self.gaussian_init_min = gaussian_init_min
        self.gaussian_init_max = gaussian_init_max
        gauss_act = lambda x: GaussAct(x, gaussian_init_min, gaussian_init_max)

        # Creates the first module of the network 
        self.model_density_1 = nn.Sequential(
            nn.Linear(3, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
        )

        # Creates the second module of the network (skip connection of input to the network again)
        self.model_density_2 = nn.Sequential(
            nn.Linear(256 + 3, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 256),
            gauss_act(256),
            nn.Linear(256, 256 + 1)
        )

        # Creates activation function for density
        self.softplus = nn.Softplus(threshold=8)

        # Creates the final module of the model that outputs the color
        self.model_color = nn.Sequential(
            nn.Linear(256 + 3, 128),
            gauss_act(128),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )


    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        z = self.model_density_1(pos)
        z = self.model_density_2(th.cat((z, pos), dim=1))

        # Extract density
        density = self.softplus(z[:, 256] - 1)

        # Extract the color estimate
        rgb = self.model_color(th.cat((z[:, :256], dir), dim=1))


        # Return estimates
        return rgb, density
