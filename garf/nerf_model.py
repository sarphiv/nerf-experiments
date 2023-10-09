import torch as th
import torch.nn as nn

from gaussian import GaussAct


class NerfModel(nn.Module):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
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
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.gauss_mean = gaussian_init_min
        self.gauss_std = gaussian_init_max

        # Creates the first module of the network 
        self.model_density_1 = self.contruct_model_density(
            3,
            self.hidden_dim,
            self.hidden_dim
        )

        # Creates the activation function for the first module
        self.gauss_act = GaussAct(self.hidden_dim, gaussian_init_min, gaussian_init_max)

        # Creates the second module of the network (skip connection of input to the network again)
        self.model_density_2 = self.contruct_model_density(
            self.hidden_dim + 3,
            self.hidden_dim,
            self.hidden_dim + 1
        )
        self.softplus = nn.Softplus(threshold=8)

        # Creates the final layer of the model that outputs the color
        self.model_color = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim//2),
            GaussAct(self.hidden_dim//2, gaussian_init_min, gaussian_init_max),
            nn.Linear(self.hidden_dim//2, 3)
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # Hn( Hn(x), x )
        z = self.model_density_1(pos)
        z = self.gauss_act(z)
        z = self.model_density_2(th.cat((z, pos), dim=1))

        density = self.softplus(z[:, self.hidden_dim] - 1)

        # H(Hn(Hn(x), x), d)
        # Extract the color estimate
        rgb = self.model_color(th.cat((z[:, :self.hidden_dim], dir), dim=1))

        # Clamp the rgb values to be between 0 and 1
        rgb = self.sigmoid(rgb)


        return density, rgb


    def contruct_model_density(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """
        Creates an MLP with n_hidden layers with the requested input and output dimensionality.
        The hidden layers have a Gaussian activation function. 
        """
        if self.n_hidden == 0:
            return nn.Linear(input_dim, output_dim)
        else:
            # The first and last layers have special dimensions
            layer_first = nn.Linear(input_dim, hidden_dim)
            layer_last = nn.Linear(hidden_dim, output_dim)
            intermediate_layers = []
            
            # Create n_hidden identical layers of gaussian activation and linear mapping
            for _ in range(self.n_hidden-1):
                intermediate_layers += [GaussAct(hidden_dim, self.gauss_mean, self.gauss_std), nn.Linear(hidden_dim, hidden_dim)]

            # Concatenate the layers into one sequential
            return nn.Sequential(
                layer_first, 
                *intermediate_layers, 
                GaussAct(hidden_dim, self.gauss_mean, self.gauss_std), 
                layer_last
            )
