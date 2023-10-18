
from gaussian import GaussAct
from torch import nn
import torch as th
import torch.nn.functional as F


def Function1(x, scale = 0.):
    f1 = lambda x: (th.sin(x/4)*4 + th.sin(2*x))*(x > 0).float()
    f2 = lambda x: th.ones_like(x)*2*(x <= 0).float()
    f1_int_ = lambda x: -th.cos(x/4)*16 - th.cos(2*x)/2
    f2_int_ = lambda x: 2*x
    f1_int = lambda x: f1_int_(x)*(x > 0).float() + f1_int_(th.tensor(0))*(x <= 0)
    f2_int = lambda x: f2_int_(x)*(x <= 0).float() + f2_int_(th.tensor(0))*(x > 0)
    F1 = lambda x,s: (f1_int(x+s/2) - f1_int(x-s/2))/s
    F2 = lambda x,s: (f2_int(x+s/2) - f2_int(x-s/2))/s

    mask = (x > 25.001) | (x < -25.001)
    if scale == 0:
        y = f1(x) + f2(x)
        y[mask] = th.nan
    else:
        y = F1(x, scale) + F2(x, scale)
        y[mask] = th.nan

    return y


def Function2(x, scale = 0., frequency = th.pi):
    # similar to Function1, but the underlying function
    # is just one sinus curve
    f = lambda x: th.sin(th.pi*x*frequency)
    f_int = lambda x: -th.cos(th.pi*x*frequency)/frequency/th.pi
    F = lambda x,s: (f_int(x+s/2) - f_int(x-s/2))/s
    mask = (x > 25.001) | (x < -25.001)
    if scale == 0:
        y = f(x)
        y[mask] = th.nan
    else:
        y = F(x, scale)
        y[mask] = th.nan
    
    return y



class MyLinear(nn.Linear):
    pass
    # def forward(self, x):
    #     return F.linear(x, self.weight/th.sum(self.weight, dim=1, keepdim=True).detach(), self.bias)

class GaussActMLP(nn.Module):
    def __init__(self, n_hidden, input_dim, output_dim, hidden_dim, initial_values):
        super().__init__()
        self.activations = []
        self.layers = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden

        for i in range(n_hidden):
            if i == 0:
                self.layers.append(MyLinear(input_dim, hidden_dim))
            else:
                self.layers.append(MyLinear(hidden_dim, hidden_dim))
            self.activations.append(GaussAct(hidden_dim, initial_values))
        self.layers.append(MyLinear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)
        
    def forward(self, x: th.Tensor, scale=0.):
        for i in range(self.n_hidden):
            x = self.layers[i](x)
            x = self.activations[i](x, (scale*(i==0 or i==self.n_hidden-1)))
        x = self.layers[-1](x)
        return x

