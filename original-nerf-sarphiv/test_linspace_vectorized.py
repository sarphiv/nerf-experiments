import torch as th

batch_size = 7
n_samples_coarse = 4
n_samples_fine = 20


t = th.rand(batch_size, n_samples_coarse)
t = th.cumsum(t, dim=1)


weights = th.rand_like(t)
weights = weights / weights.sum(dim=1, keepdim=True)

far = t.max()
t = th.hstack((t, th.ones(batch_size, 1)*t.max()+0.2))

fine_samples = th.round(weights*n_samples_fine)
fine_samples[th.arange(batch_size), th.argmax(fine_samples, dim=1)] += n_samples_fine - fine_samples.sum(dim=1)
fine_samples_cum_sum = th.hstack((th.zeros(batch_size, 1), fine_samples.cumsum(dim=1)))

arange = th.arange(n_samples_fine).unsqueeze(0)

bin_mask = th.zeros(batch_size, n_samples_fine).int()

for i in range(n_samples_coarse):
    mask = (arange >= fine_samples_cum_sum[:, i].unsqueeze(-1)) & (arange < fine_samples_cum_sum[:, i+1].unsqueeze(-1))
    bin_mask += i*mask# - fine_samples_cum_sum[:, i].unsqueeze(-1)*mask
    # bin_mask += i*mask# - fine_samples_cum_sum[:, i].unsqueeze(-1)*mask

print(bin_mask)

# print(t)
# print(t_fine)
