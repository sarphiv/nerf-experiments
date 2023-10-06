import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



class RotationMatrix(nn.Module):
    def __init__(self, param=None):
        super().__init__()
        if param is None:
            self.param = nn.Parameter(th.randn(3))
        else:
            self.param = th.tensor(param, requires_grad=False).float()

    def forward(self):
        a_helper = th.tensor([[0, -1, 0],[1, 0, 0],[0, 0, 0]], requires_grad=False).float()
        b_helper = th.tensor([[0, 0, 1],[0, 0, 0],[-1, 0, 0]], requires_grad=False).float()
        c_helper = th.tensor([[0, 0, 0],[0, 0, -1],[0, 1, 0]], requires_grad=False).float()

        a = self.param[0]*a_helper
        b = self.param[1]*b_helper
        c = self.param[2]*c_helper

        return th.matrix_exp(a+b+c)




a,b,c = 10, 6, 21

r = RotationMatrix([a,b,c])
t = th.randn(1,3)*10

O = th.randn(10,3, requires_grad=False)*10
X = O@r().T + th.randn_like(O)*2

r_hat = RotationMatrix()
optimiser = th.optim.SGD(r_hat.parameters(), lr=0.0001)

loss = []
epoch = []
error = []

for i in tqdm(range(1000)):
    optimiser.zero_grad()
    X_hat = O@r_hat().T
    l = th.mean((X-X_hat)**2)
    l.backward()
    loss.append(l.item())
    error.append(th.mean((r()-r_hat())**2).item())
    epoch.append(i)
    optimiser.step()


print(r.param)
print(r_hat.param)

print(r()-r_hat())

plt.plot(epoch, loss, label='loss')
plt.plot(epoch, error, label='error')
plt.legend()
plt.show()