import torch as th
import torch.nn as nn

from linear_sine import LinearSine


class NerfModel(nn.Module):
    def __init__(self, input_scale: float):
        super().__init__()
        
        self.input_scale = input_scale
        self.hidden_dim = 256

        self.model_density_1 = nn.Sequential(
            LinearSine(
                3, 
                self.hidden_dim, 
                scale=input_scale,
                first_layer=True
            ),
            LinearSine(self.hidden_dim, self.hidden_dim),
            LinearSine(self.hidden_dim, self.hidden_dim),
            LinearSine(self.hidden_dim, self.hidden_dim)
        )
        self.model_density_2 = nn.Sequential(
            LinearSine(
                self.hidden_dim + 3, 
                self.hidden_dim, 
                scale=th.cat((
                    th.ones(self.hidden_dim), 
                    th.ones(3) * input_scale
                ))
            ),
            LinearSine(self.hidden_dim, self.hidden_dim),
            LinearSine(self.hidden_dim, self.hidden_dim),
            # Add 3 for rgb and 1 for density
            nn.Linear(self.hidden_dim, self.hidden_dim + 3 + 1)
        )

        self.softplus = nn.Softplus(threshold=8)

        self.model_color = nn.Sequential(
            LinearSine(
                # Add 3 for direction
                self.hidden_dim + 3, 
                self.hidden_dim,
                scale=th.cat((
                    th.ones(self.hidden_dim), 
                    th.ones(3) * input_scale
                ))
            ),
            nn.Linear(self.hidden_dim, 3)
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, pos: th.Tensor, dir: th.Tensor):
        # Calculate density with skip connection
        z = self.model_density_1(pos)
        z = self.model_density_2(th.cat((z, pos), dim=1))

        # NOTE: Using shifted softplus like mip-NeRF instead of ReLU as in the original paper.
        #  The weight initialization seemed to cause negative initial values.
        density = self.softplus(z[:, (self.hidden_dim+3)] - 1)

        # NOTE: Encouraged to learn residual color
        rgb_latent = z[:, :self.hidden_dim]
        rgb_base = z[:, self.hidden_dim:(self.hidden_dim+3)]
        rgb_res = self.model_color(th.cat((rgb_latent, dir), dim=1))
        rgb = self.sigmoid(rgb_base + rgb_res)


        return density, rgb
