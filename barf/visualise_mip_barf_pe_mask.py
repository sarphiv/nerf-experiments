from math import log2

import torch as th

from model_interpolation_architecture import NerfModel
from positional_encodings import BarfPositionalEncoding, IntegratedFourierFeatures, IntegratedBarfFourierFeatures, FourierFeatures
from model_mip import MipNeRF, MipBarf


if __name__ == "__main__":

    BLUR_SIGMA = 40

    position_encoder = IntegratedFourierFeatures(
        levels=10,
        include_identity=True,
        scale=1.,
        distribute_variance=True,
    )

    direction_encoder = FourierFeatures(4)

    model_radiance = NerfModel(
        n_hidden=4,
        hidden_dim=256,
        delayed_direction=True,
        delayed_density=False,
        n_segments=2,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        learning_rate_start=5e-4,
        learning_rate_stop=1e-5,
        learning_rate_decay_end=1000
    )


    model = MipBarf(
        n_training_images=10,
        start_gaussian_sigma=BLUR_SIGMA,
        camera_learning_rate_start=1e-3,
        camera_learning_rate_stop=1e-5,
        camera_learning_rate_decay_end=100,
        max_gaussian_sigma=None,
        near_sphere_normalized=2,
        far_sphere_normalized=8,
        samples_per_ray_radiance= 256,
        samples_per_ray_proposal= 64,
        model_radiance=model_radiance,
        uniform_sampling_strategy = "equidistant",
        uniform_sampling_offset_size=-1.,
        gaussian_sigma_decay_start_step=10,
        gaussian_sigma_decay_end_step=20,
        pixel_width_follows_sigma=True,
        blur_follows_sigma=False,
    )

    import matplotlib.pyplot as plt
    samples_per_rays = [2,4,8,16, 32, 64, 128, 256]
    pixel_width = (400 / 2 / th.tan(th.tensor(0.6911112070083618) / 2)).item()**(-1)
    for samples_per_ray in samples_per_rays:
        pos = th.tensor([[1,2,3]]).float().repeat(samples_per_ray,1)
        dir = th.tensor([[1,2,3]]).float().repeat(samples_per_ray,1)
        dir = dir/th.linalg.vector_norm(dir)
        t = th.linspace(2,8,samples_per_ray+1)
        t_start = t[:-1, None]
        t_end = t[1:, None]

        new_pixel_width = 10*th.tensor(pixel_width).repeat(samples_per_ray).unsqueeze(1)*model.pixel_width_scalar

        ipe, weight, sigma_t_sq, sigma_r_sq = position_encoder.forward(pos, dir, new_pixel_width, t_start, t_end, diagnose=True)

        plt.plot(weight[:,:10], label=[str(i) for i in range(10)])
        plt.legend()
        plt.title(f"sigma_r={sigma_r_sq.max().item()**0.5}, sigma_t={sigma_t_sq.max().item()**0.5}")
        plt.savefig(f"samples_per_ray={samples_per_ray}.png", )

        plt.cla()


