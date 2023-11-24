import torch as th
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = None
        self.space_dimensions = None
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


class IdentityPositionalEncoding(PositionalEncoding):
    def __init__(self, space_dimensions: int = 3):
        super().__init__()
        self.output_dim = space_dimensions
        self.space_dimensions = space_dimensions

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"



class FourierFeatures(PositionalEncoding):
    def __init__(self, levels: int, scale: float = 2*th.pi, space_dimensions: int = 3): 
        """
        Positional encoding using Fourier features of "levels" periods. 
        
        """
        super().__init__()

        self.levels = levels
        self.scale = scale
        self.space_dimensions = space_dimensions
        self.output_dim = levels*2*space_dimensions


    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for i in [0, levels-1]

        returns:
            [cos(x), cos(2x), cos(4x) ... , cos(y), cos(2y), cos(4y) ... , cos(z), cos(2z), cos(4z) ...,
             sin(x), sin(2x), sin(4x) ... , sin(y), sin(2y), sin(4y) ... , sin(z), sin(2z), sin(4z) ...]
        """

        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"

        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(x.shape[1])
        args = x.repeat_interleave(self.levels, dim=1) * scale

        return th.hstack((th.cos(args), th.sin(args)))



class BarfPositionalEncoding(PositionalEncoding):
    def __init__(self,
                 levels: int,
                 alpha_start: float,
                 alpha_increase_start_epoch: float,
                 alpha_increase_end_epoch: float,
                 include_identity: bool = True,
                 scale: float = 2*th.pi,
                 space_dimensions: int = 3):
        """
        Positional encoding using the mask from the barf paper. 
        """
        super().__init__()
        self.levels = levels
        self.alpha_start = alpha_start
        self.output_dim = (levels*2 + include_identity)*space_dimensions
        self.alpha_increase_start_epoch = alpha_increase_start_epoch
        self.alpha_increase_end_epoch = alpha_increase_end_epoch
        self.include_identity = include_identity
        self.scale = scale
        self.space_dimensions = space_dimensions
        self.register_buffer("alpha", th.tensor(float(alpha_start)))

    def update_alpha(self, epoch: float) -> None:
        """
        Updates the alpha value of the positional encoding. 
        """

        if epoch < self.alpha_increase_start_epoch:
            alpha = self.alpha_start

        elif self.alpha_increase_start_epoch <= epoch <= self.alpha_increase_end_epoch:
            alpha = (
                self.alpha_start
                + (epoch - self.alpha_increase_start_epoch)
                * (self.levels - self.alpha_start)
                / (self.alpha_increase_end_epoch - self.alpha_increase_start_epoch)
            )

        else:
            alpha = float(self.levels)
        
        self.alpha = th.tensor(alpha, device=self.alpha.device)
    

    def compute_mask(self, alpha: th.Tensor) -> th.Tensor:
        # init zero mask
        mask = th.zeros((self.levels), device=alpha.device)

        # identify the turning point, k where 1 > alpha - k > 0 mask 
        idx_ramp = int(alpha)

        # set ones
        mask[:idx_ramp] = 1.

        # the turning point is a cosine interpolation
        if idx_ramp < self.levels:
            mask[idx_ramp] = (1 - th.cos((alpha - idx_ramp) * th.pi)) / 2

        # repeat for sin and for each channel
        mask = mask.repeat(self.space_dimensions)
    
        return mask.view(1, -1)


    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Gets the positional encoding of x for each channel.
        x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for i in [0, levels-1]

        returns:
            [cos(x), cos(2x), cos(4x) ... , cos(y), cos(2y), cos(4y) ... , cos(z), cos(2z), cos(4z) ...,
             sin(x), sin(2x), sin(4x) ... , sin(y), sin(2y), sin(4y) ... , sin(z), sin(2z), sin(4z) ...]

            if include_identity:
                [x,y,z] is prepended to the output
        """

        assert x.shape[1] == self.space_dimensions, f"Input shape {x.shape} does not match space dimensionality {self.space_dimensions}"

        scale = self.scale*(2**th.arange(self.levels, device=x.device)).repeat(self.space_dimensions)
        args = x.repeat_interleave(self.levels, dim=1) * scale

        mask = self.compute_mask(self.alpha)

        if self.include_identity:
            return th.hstack((x, mask*th.cos(args), mask*th.sin(args)))
        else:
            return th.hstack((mask*th.cos(args), mask*th.sin(args)))
        

class NerfModel(nn.Module):
    def __init__(self, 
        n_hidden: int,
        hidden_dim: int,
        delayed_direction: bool, 
        delayed_density: bool, 
        n_segments: int,
        position_encoder: PositionalEncoding,
        direction_encoder: PositionalEncoding,
        learning_rate_start: float = 5e-4,
        learning_rate_stop: float = 5e-5,
        learning_rate_decay_end: float = 0,
    ):
        """
        This is an interpolation between the original NeRF vanilla model and a naive version
        Parameters:
        -----------
            n_hidden: int - the number of hidden layers in the model
            hidden_dim: int - the dimensionality of the hidden layers
            delayed_direction: bool - if true then the direction is only feed to the network at the last layers
            n_segments: int - the number of segments of the network where the position is feed into it
        """
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.delayed_direction = delayed_direction
        self.delayed_density = delayed_density
        self.n_segments = n_segments
        self.position_encoder = position_encoder
        self.direction_encoder = direction_encoder
        self.learning_rate_start = learning_rate_start
        self.learning_rate_stop = learning_rate_stop
        self.learning_rate_decay_end = learning_rate_decay_end

        # Dimensionality of the input to the network 
        positional_dim = self.position_encoder.output_dim
        directional_dim = self.direction_encoder.output_dim

        # Create list of model segments
        self.model_segments = nn.ModuleList()
        if n_segments == 0:
            raise NotImplementedError("n_segments must be greater than 0")
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

    def forward(self,
                pos: th.Tensor,
                dir: th.Tensor, 
                ) -> tuple[th.Tensor, th.Tensor]:

        pos = self.position_encoder(pos)
        dir = self.direction_encoder(dir)

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

    def list_segments(self):
        # Iterate through the ModuleList and list its segments
        for i, segment in enumerate(self.model_segments):
            print(f"Segment {i}: {segment}")
        
        print(f"Final layer: {self.model_color}")

