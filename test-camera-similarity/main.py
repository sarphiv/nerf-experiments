from gaussian import GaussAct as GaussActGarf

import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt

from itertools import product
from typing import cast

class GaussAct(GaussActGarf):
    pass

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
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.activations.append(GaussAct(initial_values))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)
        
    def forward(self, x: th.Tensor, scale=0.):
        for i in range(self.n_hidden):
            x = self.layers[i](x)
            x = self.activations[i](x, scale)#*(i==0 or i==self.n_hidden-1))
        x = self.layers[-1](x)
        return x


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

    # plt.plot(x, f1(x), label="f1")
    # plt.plot(x, f1_int(x), label="f1_int")
    # plt.plot(x, F1(x, 1.), label="F1")
    # plt.legend()
    # plt.show()
    # plt.plot(x, f2(x), label="f2")
    # plt.plot(x, f2_int(x), label="f2_int")
    # plt.plot(x, F2(x, 1.), label="F2")
    # plt.legend()
    # plt.show()



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


def true_func(x, scale, frequency=1):
    return Function1(x, scale)
    # return Function2(x, scale, frequency)

hidden_dim = 70
n_hidden = 3
epochs = 20000
# scales_test = th.tensor([[6,2],[0.1, th.nan]])
scales_test = th.tensor([[5., 1., ],[0.5, 0.]])
size_train = lambda: int(th.randint(10, 100, (1,)).int().item())
x_train_func = lambda size: th.rand(size, 1)*40 - 20 + th.rand(size, 1)*8 - 4

initial_values_var = lambda shape: th.rand(shape)*6

model = GaussActMLP(n_hidden, 1, 1, hidden_dim, initial_values_var)


optimizer_gauss = th.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()
plt.ion()
fig, axs = plt.subplots(*scales_test.shape)#, sharex=True, sharey=True, figsize=(15,23))
axs = axs.reshape(scales_test.shape)
# x = th.linspace(-35, 35, 10000)
# ax.plot(x, true_func(x), label="true")
# ax.plot(x, true_func(x, 3.), label="true: scale=2.5")
# plt.show()
# exit()

def slice_(arr, idx):
    if len(idx) == 1:  return arr[idx[0]]
    else:              return slice_(arr[idx[0]], idx[1:])

for j in range(epochs):
    optimizer_gauss.zero_grad()
    size_train_epoch = size_train()
    # size_train = 1000
    x_train = x_train_func(size_train_epoch)
    scale = 50/size_train_epoch
    y_train = true_func(x_train, scale)
    y_pred = model(x_train, scale)
    # loss_activation = sum([a.loss for a in model.activations])
    loss = loss_func(y_pred, y_train)# + loss_activation
    loss.backward()
    optimizer_gauss.step()
    if j % 100 == 0:
        with th.no_grad():
            x_test = th.linspace(-35, 35, 1000).reshape(-1, 1)
            for idx in product(*[range(i) for i in axs.shape]):
                ax = cast(plt.Axes, slice_(axs, idx))
                ax.clear()
                scale = cast(th.Tensor, slice_(scales_test, idx))
                if th.isnan(scale):
                    act_params = th.cat([a.variance for a in model.activations]).flatten() #type: ignore
                    ax.hist(act_params, bins=50)
                else:
                    y_test = true_func(x_test, cast(float, scale))
                    y_pred_test1 = model(x_test, scale)
                    ax.plot(x_test, y_test, label="true")
                    ax.plot(x_test, y_pred_test1, label="pred")
                    ax.set_title(f"scale={scale:.3}")
                    ax.legend()

            # ax.clear()
            # # ax.plot(x_test2, y_test2, label="true1")
            # # ax.plot(x_test2, y_pred_test2, label="pred")
            # mask = th.argsort(x_train.flatten())
            # ax.plot(x_train[mask], y_train[mask], label="true")
            # ax.plot(x_train[mask], y_pred[mask], label="pred")
            # ax.set_title(f"{scale}")
            # ax.legend()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            print(f"Epoch {j}: loss_gauss = {loss}")



plt.savefig("fig.png", dpi=300)