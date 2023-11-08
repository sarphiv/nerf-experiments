from gaussian import GaussAct
from model import GaussActMLP, Function1, Function2

import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt

from itertools import product
from typing import cast


def true_func(x, scale, frequency=1):
    return Function1(x, scale)
    # return Function2(x, scale, frequency)

hidden_dim = 70
n_hidden = 3
epochs = 20000
# scales_test = th.tensor([[6,2],[0.1, th.nan]])
scales_test = th.tensor([[5., 2.5, ],[0.5, th.nan]])
size_train = lambda: int(th.randint(10, 100, (1,)).int().item())
x_train_func = lambda size: th.rand(size, 1)*40 - 20 + th.rand(size, 1)*8 - 4

initial_values_func = lambda shape: th.rand(shape)*10 + 0.5
model = GaussActMLP(n_hidden, 1, 1, hidden_dim, initial_values_func)

optimizer_gauss = th.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()
plt.ion()
fig, axs = plt.subplots(*scales_test.shape)#, sharex=True, sharey=True, figsize=(15,23))
axs = axs.reshape(scales_test.shape)

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
                    act_params = th.cat([a.param for a in model.activations]).flatten() #type: ignore
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