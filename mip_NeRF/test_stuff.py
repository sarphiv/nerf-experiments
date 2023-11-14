
# import torch as th
# from torch import Tensor as T, tensor as t
# import pytorch_lightning as pl



# class Dummy(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.par1 = th.nn.Parameter(th.tensor([[1.,2]]))
#         self.par2 = th.nn.Parameter(th.tensor(3.))
#         self.automatic_optimization=False
    
#     def forward(self, x: T):
#         x =  (x * self.par1).sum(dim=1)
#         x = x + self.par2
#         return x

#     def configure_optimizers(self):
#         self.optimizer1 = th.optim.Adam(params = iter([self.par1]), lr=0.001)
#         self.optimizer2 = th.optim.Adam(params = iter([self.par2]), lr=0.001)
#         return [self.optimizer1, self.optimizer2]
    
#     #getter function for the gradient

#     @property
#     def grad(self):
#         return self.par1.grad, self.par2.grad
    
#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         loss = th.nn.functional.mse_loss(y,self(x))


#         self.optimizer1.zero_grad()
#         self.manual_backward(loss, retain_graph=True)
#         self.optimizer1.step()

#         self.optimizer2.zero_grad()
#         self.manual_backward(loss, retain_graph=True)
#         self.optimizer2.step()


# d = Dummy()

# trainer = pl.Trainer()



# x = th.tensor([[1,2.],[3,8.],[1,0.],[89,1.], [2,2],[1,2], [89,89], [89,-2]])

# par = th.tensor([[2,17.]])

# with th.no_grad():
#     y = d.forward(x)
#     y = y + th.randn_like(y)


# dataloader = th.utils.data.DataLoader(th.utils.data.TensorDataset(x,y), batch_size=4)

# trainer.fit(d, dataloader, dataloader)

# d.training_step((x,y), None)


import torch as th
from torch import Tensor as T, tensor as t
# import pytorch_lightning as pl

N = 10

idx = t([0,3,3,2,1,2,0,1,0,2])

noise = th.randn((4, 3, 1))*0.3

rotation_noise1 = th.matrix_exp(th.cross(
    -th.eye(3).view(1, 3, 3), 
    noise[idx],
    dim=1
))
rotation_noise2 = th.matrix_exp(th.cross(
    -th.eye(3).view(1, 3, 3), 
    noise,
    dim=1
))[idx]

print("hej")