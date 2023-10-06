from gaussian import GaussAct as GaussActGarf

import torch.nn as nn
import torch as th
import matplotlib.pyplot as plt


class GaussAct(GaussActGarf):
    def forward(self, x: th.Tensor, scale: float):
        return self.func(x, self.variance*scale)

class GaussActMLP(nn.Module):
    def __init__(self, n_hidden, input_dim, output_dim, hidden_dim):
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
            self.activations.append(GaussAct(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(self.layers)
        self.activations = nn.ModuleList(self.activations)
        
    def forward(self, x: th.Tensor, scale: float):
        for i in range(self.n_hidden):
            x = self.layers[i](x)
            x = self.activations[i](x, scale if i==self.n_hidden-1 else 1.)
        x = self.layers[-1](x)
        return x



true_func = lambda x: th.sin(x/4)*4 + th.sin(2*x)

input_dim = 1
output_dim = 1
hidden_dim = 100
n_hidden = 3
epochs = 1000

model = GaussActMLP(n_hidden, input_dim, output_dim, hidden_dim)

optimizer = th.optim.Adam(model.parameters(), lr=0.004)
loss_func = nn.MSELoss()

for i in range(epochs):
    optimizer.zero_grad()
    x = th.rand(1000, input_dim)*50 - 25
    y = true_func(x)
    y_pred = model(x, 1.)
    loss = loss_func(y_pred, y)
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(loss.item())

pred_func = lambda x, scale: model(x, scale).detach().numpy()

x = th.linspace(-25, 25, 1000).reshape(-1, 1)

plt.plot(x, true_func(x), label="true")
plt.plot(x, pred_func(x, 1.), label="pred")
plt.plot(x, pred_func(x, 4.), label="pred4")

plt.legend()
plt.show()