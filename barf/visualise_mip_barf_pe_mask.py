from math import log2

import torch as th

from model_interpolation_architecture import NerfModel
from positional_encodings import BarfPositionalEncoding, IntegratedFourierFeatures, IntegratedBarfFourierFeatures, FourierFeatures
from model_mip import MipNeRF, MipBarf


if __name__ == "__main__":

    BLUR_SIGMA = 100

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


    def get_model(start_pixel_width_sigma, start_blur_sigma):
        return MipBarf(
            n_training_images=10,
            camera_learning_rate_start=1e-3,
            camera_learning_rate_stop=1e-5,
            camera_learning_rate_decay_end=100,
            near_sphere_normalized=2,
            far_sphere_normalized=8,
            samples_per_ray_radiance= 256,
            samples_per_ray_proposal= 64,
            model_radiance=model_radiance,
            uniform_sampling_strategy = "equidistant",
            uniform_sampling_offset_size=-1.,
            sigma_decay_start_step=10,
            sigma_decay_end_step=20,
            start_blur_sigma=start_blur_sigma,
            start_pixel_width_sigma=start_pixel_width_sigma,

        )

    import matplotlib.pyplot as plt
    samples_per_rays = [128]
    pixel_width = (400 / 2 / th.tan(th.tensor(0.6911112070083618) / 2)).item()**(-1)
    for samples_per_ray in samples_per_rays:
        for start_pixel_width_sigma in range(0,50,5):
            pos = th.tensor([[1,2,3]]).float().repeat(samples_per_ray,1)
            dir = th.tensor([[1,2,3]]).float().repeat(samples_per_ray,1)
            dir = dir/th.linalg.vector_norm(dir)
            t = th.linspace(2,8,samples_per_ray+1)
            t_start = t[:-1, None]
            t_end = t[1:, None]
            model = get_model(start_pixel_width_sigma, 10000)
            new_pixel_width = 10*th.tensor(pixel_width).repeat(samples_per_ray).unsqueeze(1)

            ipe, weight, sigma_t_sq, sigma_r_sq = position_encoder.forward(pos, dir, new_pixel_width, t_start, t_end, diagnose=True)

            plt.plot(weight[:,0], label=f"{start_pixel_width_sigma}")
        plt.legend()
        plt.title(f"sigma={start_pixel_width_sigma}_sigma_r={sigma_r_sq.max().item()**0.5:.3f}, sigma_t={sigma_t_sq.max().item()**0.5:.3f}")
        plt.savefig(f"samples_per_ray={samples_per_ray}.png", )

        plt.cla()


