import torch as th
from mip_model import IntegratedFourierFeatures
from model_interpolation_architecture import FourierFeatures
import matplotlib.pyplot as plt

# th.manual_seed(0)

pixel_width = 1.

t = th.rand(10).sort()[0]*2*th.pi
t_start = t[:-1].view(-1, 1)
t_end = t[1:].view(-1, 1)



orig = th.randn(1,3)*3 + 5
dir = th.randn(1,3)
dir = dir / th.sum(dir**2, dim=1, keepdim=True)**0.5
dir = dir.repeat(9, 1)


pos = orig + dir * t_start

ipe_encoder = IntegratedFourierFeatures(4, 1.)
pe_encoder = FourierFeatures(4, 1.)

ipe = ipe_encoder(pos, dir, t_start, t_end, pixel_width)
pe = pe_encoder(pos)

