
import torch as th
from torch import Tensor as T, tensor as t
import pytorch_lightning as pl



class Dummy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.par1 = th.nn.Linear(2,2)
        self.par2 = th.nn.Linear(2,2)
        self.par3 = th.nn.Linear(2,1)
        self.automatic_optimization=False
    
    def forward(self, x: T):
        x = th.nn.functional.relu(self.par1(x))
        x = th.nn.functional.relu(self.par2(x))
        z = self.par3(x)
        return z

    def configure_optimizers(self):
        self.optimizer1 = th.optim.Adam(params = self.par1.parameters(), lr=0.001)
        self.optimizer2 = th.optim.Adam(params = self.par2.parameters(), lr=0.001)
        self.optimizer3 = th.optim.Adam(params = self.par3.parameters(), lr=0.001)
        return [self.optimizer1, self.optimizer2, self.optimizer3]
    
    #getter function for the gradient

    @property
    def grad(self):
        return self.par1.weight.grad, self.par2.weight.grad, self.par3.weight.grad
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = th.nn.functional.mse_loss(y,self(x))

        self.optimizer3.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        self.optimizer3.step()

        self.optimizer1.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        self.optimizer2.step()



d = Dummy()

trainer = pl.Trainer()



x = th.tensor([[1,2.],[3,8.],[1,0.],[89,1.], [2,2],[1,2], [89,89], [89,-2]])

par = th.tensor([[2,17.]])

with th.no_grad():
    y = d.forward(x)
    y = y + th.randn_like(y)


dataloader = th.utils.data.DataLoader(th.utils.data.TensorDataset(x,y), batch_size=4)

trainer.fit(d, dataloader, dataloader)

d.training_step((x,y), None)


# import torch as th
# from torch import Tensor as T, tensor as t
# # import pytorch_lightning as pl

# def compute_mask(levels, alpha: th.Tensor) -> th.Tensor:
#     # init zero mask
#     mask = th.zeros((levels))

#     # identify the turning point, k where 1 > alpha - k > 0 mask 
#     idx_ramp = int(alpha)

#     # set ones
#     mask[idx_ramp:] = 1.

#     # the turning point is a cosine interpolation
#     if idx_ramp < levels:
#         mask[idx_ramp] = (1 - th.cos((alpha - idx_ramp) * th.pi)) / 2

#     return mask.view(1, -1)


# def forward(levels, x: th.Tensor, alpha) -> th.Tensor:
#     """
#     Gets the positional encoding of x for each channel.
#     x_i in [-0.5, 0.5] -> function(x_i * pi * 2^j) for function in (cos, sin) for i in [0, levels-1]

#     returns:
#         [cos(x), cos(2x), cos(4x) ... , cos(y), cos(2y), cos(4y) ... , cos(z), cos(2z), cos(4z) ...,
#             sin(x), sin(2x), sin(4x) ... , sin(y), sin(2y), sin(4y) ... , sin(z), sin(2z), sin(4z) ...]

#         if include_identity:
#             [x,y,z] is prepended to the output
#     """

#     scale = 2*th.pi*(2**th.arange(levels, device=x.device)).repeat(x.shape[1])
#     args = x.repeat_interleave(levels, dim=1) * scale

#     mask = compute_mask(levels, alpha).repeat(1,3)

#     if True:
#         return th.hstack((x, mask*th.cos(args), mask*th.sin(args)))
#     else:
#         return th.hstack((mask*th.cos(args), mask*th.sin(args)))
    
# print(th.pi)


# # alpha_decay_start_epoch = 1.3
# # alpha_decay_end_epoch = 8.7
# # alpha_start = 0.4
# # levels = t(6.)

# # for epoch in th.linspace(0,10,100):

# #     if alpha_decay_start_epoch > epoch:
# #         alpha = alpha_start

# #     elif alpha_decay_start_epoch <= epoch <= alpha_decay_end_epoch:
# #         alpha = (
# #             alpha_start
# #             + (epoch - alpha_decay_start_epoch)
# #             * (levels - alpha_start)
# #             / (alpha_decay_end_epoch - alpha_decay_start_epoch)
# #         )
    
# #     else: alpha = levels

# #     print(epoch, alpha)