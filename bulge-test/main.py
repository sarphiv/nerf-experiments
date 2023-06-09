import numpy as np
import matplotlib.pyplot as plt



def to_homo(ps):
    return np.vstack((ps, np.ones(ps.shape[1])))
    
def to_inho(qs):
    return (qs / qs[-1, :])[:-1]


n = 2
f = 3
r = 1
t = 1

M = np.array([
    [n/r, 0, 0, 0],
    [0, n/t, 0, 0],
    [0, 0, (f+n)/(f-n), 2*f*n/(f-n)],
    [0, 0, -1, 0]
])






zs = np.linspace(-2, -1, 100)
ys = 0
xs = np.linspace(-0.5, 0.5, 100)

# points - normal world coordinates
# points.shape = [3, n_points]
ps = [[x,ys,zs[0]] for x in xs] + [[x,ys,zs[-1]] for x in xs] + [[xs[0],ys,z] for z in zs] + [[xs[-1],ys,z] for z in zs]
ps = np.array(ps).T

def plot(points):
    n = points.shape[1]
    for i in range(10):
        lower = int(n/10*i)
        upper = int(n/10*(i+1))
        plt.scatter(points[2, lower:upper], points[0, lower:upper], s=0.4, label=i)


# Create 1x2 plot
fig = plt.figure()
fig.add_subplot(1,3,1)
plot(ps)

ps_project = to_inho(M @ to_homo(ps))

fig.add_subplot(1,3,2)
plot(ps_project)


fig.add_subplot(1,3,3)
ps_real = np.copy(ps)
ps_real[0, :] *= -n/ps[-1]/t
ps_real[1, :] *= -n/ps[-1]/r

# plt.scatter(ps_real[2, :], ps_real[0, :], s=0.2)
plot(ps_real)
plt.show()


print((ps_real[0] - ps_project[0]).max())
plt.hist(ps_real[0] - ps_project[0])
plt.show()