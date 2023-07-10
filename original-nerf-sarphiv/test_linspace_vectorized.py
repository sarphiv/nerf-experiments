import torch as th



class Self:
    def __init__(self, samples_per_ray_coarse, samples_per_ray_fine):
        self.samples_per_ray_coarse = samples_per_ray_coarse
        self.samples_per_ray_fine = samples_per_ray_fine

self = Self(4, 20)
batch_size = 7
self.samples_per_ray_coarse = 4
self.samples_per_ray_fine = 20


t_coarse = th.rand(batch_size, self.samples_per_ray_coarse)
t_coarse = th.cumsum(t_coarse, dim=1)


weights = th.rand_like(t_coarse)
weights = weights / weights.sum(dim=1, keepdim=True)

far = t_coarse.max()
t_coarse = th.hstack((t_coarse, th.ones(batch_size, 1)*far+0.2))

distances_coarse = t_coarse[:, 1:] - t_coarse[:, :-1]

device = t_coarse.device


fine_samples = th.round(weights*self.samples_per_ray_fine)
fine_samples[th.arange(batch_size), th.argmax(fine_samples, dim=1)] += self.samples_per_ray_fine - fine_samples.sum(dim=1)
fine_samples += 1
fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1, device=device), fine_samples.cumsum(dim=1)))

arange = th.arange(self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device).unsqueeze(0)
t_fine = th.zeros(batch_size, self.samples_per_ray_fine + self.samples_per_ray_coarse, device=device)


for i in range(self.samples_per_ray_coarse):
    mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
    t_fine += t_coarse[:, i].unsqueeze(-1)*mask
    t_fine += (arange - fine_samples_cum_sum[:, i].unsqueeze(-1))*mask*distances_coarse[:, i].unsqueeze(-1)/fine_samples[:, i].unsqueeze(-1)

print(t_fine)
