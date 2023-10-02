import torch as th
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

def iterative_optimize(a_true, x, t, reg=0, lr=0.001, max_iter=1000, verbose=True):

    class SimpleSimple(nn.Module):
        def __init__(self, a_true):
            super().__init__()
            self.a = nn.Parameter(th.randn_like(a_true))
            # self.p = nn.Parameter(th.randn(1))

        
        def forward(self, x):
            # t = th.tensor([[0,1],[-1,0]], requires_grad=False).float()
            # tmp = th.exp(t*self.p)
            # self.a = tmp.detach().clone()
            # return x@tmp
            return x@self.a

    th.manual_seed(1)


    s = SimpleSimple(a_true)
    optim = th.optim.SGD(s.parameters(), lr=lr)

    loss = []
    epoch = []
    params = []
    x_pred = []
    rot_losss = []


    for i in range(max_iter):
        optim.zero_grad()
        pred = s(x)
        l = th.mean((t - pred)**2) + reg*th.norm(s.a.T@s.a-th.eye(len(m)))**2
        rot_loss = th.sum((s.a.T @ s.a - th.eye(m[1]))**2)
        if reg > 0:
            l += rot_loss
        l.backward()

        x_pred.append(pred.detach().numpy())
        loss.append(l.item()/x.shape[0])
        epoch.append(i)
        params.append(s.a.detach().numpy().copy())
        rot_losss.append(rot_loss.item())

        optim.step()
        if l.item() < 0.001 + empirical_noise_level: break

    # x_pred.append(pred.detach().numpy())
    # loss.append(l.item()/x.shape[0])
    # epoch.append(i)
    # params.append(s.a.detach().numpy())

    if verbose:

        x_pred = np.array(x_pred)
        params = np.array(params).reshape(-1, n_param)

        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.plot(epoch, np.sqrt(loss), label='loss')
        plt.legend()
        fig.add_subplot(2,2,2)
        plt.plot(epoch, empirical_noise_level*np.ones_like(epoch), label='empirical noise level')
        for i in range(np.prod(m)):
            plt.plot(epoch, params[:,i], label=f'param {i}')
        plt.legend()
        fig.add_subplot(2,2,3)
        # for i in range(5):
        #     plt.plot(x_pred[:,i,0], x_pred[:,i,1], label=f'pred {i}')
        mask = np.argsort(x[:,0])[:4]
        plt.plot(x[mask,0], x[mask,1], "-o", label='start')
        plt.plot(t[mask,0], t[mask,1], "-o", label='true')
        plt.plot(x_pred[-1,mask,0], x_pred[-1,mask,1], "-o", label='pred')
        lin = np.linspace(0,2*np.pi,100)
        plt.plot(np.cos(lin)*10, np.sin(lin)*10, label='circle')
        plt.axis('equal')


        # plt.legend()
        fig.add_subplot(2,2,4)
        plt.plot(epoch, rot_losss, label='rot loss')
        # plt.legend()


        sa = s.a.detach().numpy()
        print("predicted A", sa, sep='\n')
        print("true A", a_true, sep='\n')
        print("predicted A^TA", sa.T@sa, sep='\n')

        plt.show()
    
    return s.a.detach().numpy()


def align_rotation(P, Q):
    """
    optimize ||P@R - Q||^2 using SVD
    """
    H = P.T@Q
    U, S, V = th.linalg.svd(H)
    d = th.linalg.det(V@U.T)
    K = th.eye(len(S))
    K[-1,-1] = d
    R = U@K@V.T
    return R
    

def align_paired_point_clouds(P, Q):
    """
    align paired point clouds P and Q
    by translating and rotating P into Q
    """
    # translate P to origin
    cP = th.mean(P, dim=0, keepdim=True)
    cQ = th.mean(Q, dim=0, keepdim=True)
    # rotate P to Q
    R = align_rotation(P-cP, Q-cQ)
    Qhat = (P - cP)@R + cQ
    return Qhat, R, cQ@R - cP # t = cQ@R - cP



a1 = 2
a2 = 1
noise_level = 0.5
N=6
translate = th.tensor([4,-3], requires_grad=False)*4

a_true = np.array([[a1,a2], [-a2,a1]])
a_true = th.tensor(a_true, requires_grad=False).float()
a_true = a_true / th.norm(a_true, dim=0)

P = th.randn((N, 2), requires_grad=False)
P[:, 1] *= 4.
Q = P@a_true
noise = th.randn_like(P, requires_grad=False)*noise_level
Q += noise + translate

def plot(X, **kwargs):
    plt.plot(X[:,0], X[:,1], "o", **kwargs)

plot(P, label='X')
plot(Q, label='Y')


R = align_rotation(P, Q)
Qhat, R_, t = align_paired_point_clouds(P, Q)

plot(Qhat, label='X*R+t')

# plot(P - cP, label='Pbar')
# plot(Q - cQ, label='Qbar')
# plot((P - cP)@R.T, label='PbarR')

plt.legend()
plt.axis('equal')
plt.show()

# plt.plot(t_pred[:,0], t_pred[:,1], "o", label='pred t')
# plt.legend()

# plt.show()

# print()

# print("predicted t", t_pred, sep='\n')
# print("true t", t, sep='\n')

# print(direct_optimize_rotation(x, t))
# print(a_true)
