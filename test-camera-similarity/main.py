# import torch as th
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt

# # def rot_mat(angle, idx, requires_grad=False, **kwargs):
# #     R = th.zeros((3,3), requires_grad=requires_grad, **kwargs)
# #     R[idx[0], idx[0]] = R[idx[1], idx[1]] = th.cos(angle)
# #     R[idx[1], idx[0]] = th.sin(angle)
# #     R[idx[0], idx[1]] = -th.sin(angle)
# #     i = 3 - sum(idx)
# #     R[i,i] = 1
# #     return R 

# # class RotationOptimizerExp(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         self.params = nn.Parameter(th.randn(3))

    
# #     def forward(self, O):
# #         """
# #         O: (N,3)
# #         """

# #         A = th.tensor([[0,1,0],[-1,0,0],[0,0,0]], requires_grad=False)
# #         B = th.tensor([[0,0,1], [0,0,0], [-1,0,0]], requires_grad=False)
# #         C = th.tensor([[0,0,0], [0,0,1], [0,-1,0]], requires_grad=False)
# #         R = self.params[0]*A + self.params[1]*B + self.params[2]*C

# #         return R @ O.T

# # class RotationOptimizerRegularized(nn.Module):
# #     def __init__(self, alpha):
# #         super().__init__()
# #         self.alpha = alpha
# #         self.R = nn.Parameter(th.randn(3))

# #     def forward(self, O):
# #         return self.R @ O.T
        



# # N = 100

# # t_noise = 0
# # r_noise = 0.1



# # angles = th.rand(3, requires_grad=False)*2*th.pi

# # R = rot_mat(angles[0], [0,1])@rot_mat(angles[1], [0,2])@rot_mat(angles[2], [1,2])

# # O = th.randn(N, 3, requires_grad=False)
# # C = R @ O.T + th.rand_like(O.T, requires_grad=False)*t_noise


# # ro_exp = RotationOptimizerExp()
# # ro_reg = RotationOptimizerRegularized(2.)
# # optimiser = th.optim.SGD(ro_reg.parameters(), lr=0.001)
# # mse = th.nn.MSELoss()


# # ro = ro_reg
# # criterion = mse

# # for i in range(10000):
# #     optimiser.zero_grad()

# #     C_hat = ro_reg(O)
# #     l = criterion(C_hat, C)
# #     l.backward()
# #     optimiser.step()
# #     if not i%100: print(l.item())


# class SimpleSimple(nn.Module):
#     def __init__(self, a_true):
#         super().__init__()
#         self.a = nn.Parameter(th.randn_like(a_true))
#         # self.p = nn.Parameter(th.randn(1))

    
#     def forward(self, x):
#         # t = th.tensor([[0,1],[-1,0]], requires_grad=False).float()
#         # tmp = th.exp(t*self.p)
#         # self.a = tmp.detach().clone()
#         # return x@tmp
#         return x@self.a

# th.manual_seed(1)

# a1 = -4
# a2 = 1
# noise_level = 3
# reg = 10
# N=11

# a_true = np.array([[a1,a2], [-a2,a1]])
# a_true = th.tensor(a_true, requires_grad=False).float()
# a_true = a_true / th.norm(a_true, dim=0)
# m = a_true.shape
# n_param = np.prod(m)

# x = th.rand((N, m[-1]), requires_grad=False)*10.
# t = x@a_true
# noise = th.randn_like(t, requires_grad=False)*noise_level
# t += noise 
# empirical_noise_level = th.sqrt(th.mean(noise**2)).item()

# s = SimpleSimple(a_true)
# optim = th.optim.SGD(s.parameters(), lr=0.001)

# loss = []
# epoch = []
# params = []
# x_pred = []
# rot_losss = []


# for i in range(1000):
#     optim.zero_grad()
#     pred = s(x)
#     l = th.mean((t - pred)**2) + reg*th.norm(s.a.T@s.a-th.eye(len(m)))**2
#     rot_loss = th.sum((s.a.T @ s.a - th.eye(m[1]))**2)
#     if reg > 0:
#         l += rot_loss
#     l.backward()

#     x_pred.append(pred.detach().numpy())
#     loss.append(l.item()/x.shape[0])
#     epoch.append(i)
#     params.append(s.a.detach().numpy().copy())
#     rot_losss.append(rot_loss.item())

#     optim.step()
#     if l.item() < 0.001 + empirical_noise_level: break

# # x_pred.append(pred.detach().numpy())
# # loss.append(l.item()/x.shape[0])
# # epoch.append(i)
# # params.append(s.a.detach().numpy())

# x_pred = np.array(x_pred)
# params = np.array(params).reshape(-1, n_param)

# fig = plt.figure()
# fig.add_subplot(2,2,1)
# plt.plot(epoch, np.sqrt(loss), label='loss')
# plt.legend()
# fig.add_subplot(2,2,2)
# plt.plot(epoch, empirical_noise_level*np.ones_like(epoch), label='empirical noise level')
# for i in range(np.prod(m)):
#     plt.plot(epoch, params[:,i], label=f'param {i}')
# plt.legend()
# fig.add_subplot(2,2,3)
# # for i in range(5):
# #     plt.plot(x_pred[:,i,0], x_pred[:,i,1], label=f'pred {i}')
# mask = np.argsort(x[:,0])[:4]
# plt.plot(x[mask,0], x[mask,1], "-o", label='start')
# plt.plot(t[mask,0], t[mask,1], "-o", label='true')
# plt.plot(x_pred[-1,mask,0], x_pred[-1,mask,1], "-o", label='pred')
# lin = np.linspace(0,2*np.pi,100)
# plt.plot(np.cos(lin)*10, np.sin(lin)*10, label='circle')
# plt.axis('equal')


# # plt.legend()
# fig.add_subplot(2,2,4)
# plt.plot(epoch, rot_losss, label='rot loss')
# # plt.legend()


# sa = s.a.detach().numpy()
# print("predicted A", sa, sep='\n')
# print("true A", a_true, sep='\n')
# print("predicted A^TA", sa.T@sa, sep='\n')

# plt.show()
