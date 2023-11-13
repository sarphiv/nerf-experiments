import torch as th
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, levels: int, scale: float = 2*th.pi): 
        """
        Positional encoding using Fourier features of "levels" periods. 
        
        """
        super().__init__()

        self.levels = levels
        self.scale = scale


    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for j in [0, levels-1]
        """
        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(x.shape[1])
        args = x.repeat_interleave(self.levels, dim=1) * scale

        return th.hstack((th.cos(args), th.sin(args)))


class NerfModel(nn.Module):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
        fourier: tuple[bool, int, int, bool, float, float],
        delayed_direction: bool, 
        delayed_density: bool, 
        n_segments: int
    ):
        """
        This is an interpolation between the original NeRF vanilla model and a naive version
        Parameters:
        -----------
            n_hidden: int - the number of hidden layers in the model
            hidden_dim: int - the dimensionality of the hidden layers
            fourier: tuple[bool, int, int, bool, float, float] - whether to use fourier encoding and barf mask
                1: use positional encoding or not
                2,3:  number of levels for position and direction.
                4: The last bool is to determine if the barf weighting scheme is to be applied
                5: how many frequency levels to include for each epoch
                6: how many frequency levels to include initially.
            delayed_direction: bool - if true then the direction is only feed to the network at the last layers
            n_segments: int - the number of segments of the network where the position is feed into it
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.delayed_direction = delayed_direction
        self.delayed_density = delayed_density
        self.n_segments = n_segments
        self.fourier, self.fourier_levels_pos, self.fourier_levels_dir, self.barf_weight, self.fourier_levels_per_epoch, self.fourier_levels_start = fourier
        
        # If there is a positional encoding define it here
        if self.fourier:
            self.position_encoder = FourierFeatures(self.fourier_levels_pos, 2*th.pi)
            self.direction_encoder = FourierFeatures(self.fourier_levels_dir, 1.0)
            self.alpha = self.fourier_levels_start

            # Dimensionality of the input to the network 
            positional_dim = self.fourier_levels_pos*2*3
            directional_dim = self.fourier_levels_dir*2*3
        else: 
            positional_dim = 3
            directional_dim = 3 
        
        # Create list of model segments
        self.model_segments = nn.ModuleList()
        for i in range(self.n_segments):
            # Create the model segment
            input_size = positional_dim + (not self.delayed_direction)*directional_dim + (i>0)*self.hidden_dim
            model_segment = self.contruct_model_density(input_size,
                                                        self.hidden_dim,
                                                        self.hidden_dim + (not self.delayed_density)*(i == self.n_segments-1))
            
            # Add the model segment to the list of model segments
            self.model_segments.append(model_segment)
        
        # Creates the final layer of the model that outputs the color
        self.model_color = nn.Sequential(
            nn.Linear(self.hidden_dim + self.delayed_direction*directional_dim, self.hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim//2, 3 + self.delayed_density)
        )
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus(threshold=8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, pos: th.Tensor, dir: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # Apply positional encoding
        if self.fourier:
            # Apply fourier encoding and weight 
            pos = self.position_encoder(pos)
            dir = self.direction_encoder(dir)

            # If activated the barf mask for weighing positional encodings is applied 
            if self.barf_weight:
                pos *= self._get_mask(self.fourier_levels_pos, device=pos.device).unsqueeze(0) #type: ignore
                dir *= self._get_mask(self.fourier_levels_dir, device=dir.device).unsqueeze(0) #type: ignore
        
        # Apply the model segments with relu activation between segments 
        z = th.zeros((pos.shape[0], 0), device=pos.device)
        for i, model_segment in enumerate(self.model_segments):
            # Concatenate the input to the model segment
            if (not self.delayed_direction): 
                z = th.cat((z, dir), dim=1)
            z = model_segment(th.cat((z, pos), dim=1))
            if i < self.n_segments-1:
                z = self.relu(z)


        # If the density is delayed then the last value is the density and should not be used
        length = z.shape[1] 
        if not self.delayed_density:
            length = length - 1

        # Apply the final layer of the model 
        if self.delayed_direction:
            final_input = th.cat((z[:, :length], dir), dim=1)
        else: 
            final_input = z[:, :length]
        final_output = self.model_color(final_input)

        # If the density is delayed, extract the density from the final output
        if self.delayed_density:
            density = final_output[:, -1] 
        else:
            density = z[:, -1]

        # Apply softplus to the density and sigmoid to the rgb values
        density = self.softplus(density) 
        rgb = self.sigmoid(final_output[:, :3])


        return density, rgb

    def contruct_model_density(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """
        Creates a FFNN with n_hidden layers and fits the input and output dimensionality.
        The activation function is ReLU 
        """
        if self.n_hidden == 0:
            return nn.Linear(input_dim, output_dim)
        else:
            # The initial and final layers have special dimensions
            layer1 = nn.Linear(input_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, output_dim)
            intermediate_layers = []
            
            # Create n_hidden identical layers 
            for _ in range(self.n_hidden-1):
                intermediate_layers += [nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim)]

            # Concatenate the layers into one sequential
            return nn.Sequential(layer1, *intermediate_layers, nn.ReLU(True), layer2)

    def _get_mask(self, levels: int, device: str) -> th.Tensor:
        """
        Get the mask from the barf paper that fits the positional encodings
        """
        def get_mask(alpha: th.Tensor, k: th.Tensor) -> th.Tensor:
            """
            Calculates the mask from the barf paper given alpha and k. 

            Args:
                alpha (float): value that is proporitional to the batch
                k (float):     the level of the positional encoding
            
            Returns:
                float: 0 if alpha < k, 1 if alpha - k >= 1 and a cosine interpolation otherwise
            """
            result = th.zeros_like(alpha)
            
            condition1 = alpha - k < 0
            condition2 = (0 <= alpha - k) & (alpha - k < 1)
            
            result[condition1] = 0
            result[condition2] = (1 - th.cos((alpha[condition2] - k[condition2]) * th.pi)) / 2
            result[~(condition1 | condition2)] = 1
            
            return result
        
        # Create a vector of alpha values
        alpha = th.ones((levels), device=device) * self.alpha
        k = th.arange(levels, device=device)

        # Get the mask vector 
        mask = get_mask(alpha, k)

        # Reshape mask to take in (x,y,z) and repeat an extra time for sin/cos 
        mask = mask.repeat_interleave(3)
        mask = mask.repeat(2) 

        return mask


    def list_segments(self):
        # Iterate through the ModuleList and list its segments
        for i, segment in enumerate(self.model_segments):
            print(f"Segment {i}: {segment}")
        
        print(f"Final layer: {self.model_color}")

