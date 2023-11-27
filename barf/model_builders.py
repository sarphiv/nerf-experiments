from math import log2
import argparse

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from point_logger import LogCameraExtrinsics
from epoch_fraction_logger import LogEpochFraction
from model_camera_calibration import CameraCalibrationModel
from model_interpolation_architecture import NerfModel
from positional_encodings import BarfPositionalEncoding, IntegratedFourierFeatures
from model_interpolation import NerfInterpolationOurs, NerfInterpolationNerfacc, uniform_sampling_strategies, integration_strategies
from model_barf import BarfModel
from model_mip import MipNeRF, MipBarf


def nerf_interpolation_builder(
    fourier_levels_pos = 10,
    fourier_levels_dir = 4,
    include_identity = False,
    samples_per_ray_radiance = 256,
    samples_per_ray_proposal = 64,
    near_sphere_normalized = 2.,
    far_sphere_normalized = 8.,
    scale = 1.,
    n_hidden=4,
    hidden_dim=256,
    delayed_direction=True,
    delayed_density=False,
    n_segments=2,
    learning_rate_start=5e-4,
    learning_rate_stop=1e-4,
    learning_rate_decay_end=200000,
    uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
    uniform_sampling_offset_size: float = 0,
    integration_strategy: integration_strategies = "left",
):

    position_encoder = BarfPositionalEncoding(levels=fourier_levels_pos,
                                                # alpha_start=0,
                                                alpha_start=fourier_levels_pos,
                                                alpha_increase_start_epoch=1.28,
                                                alpha_increase_end_epoch=6.4,
                                                include_identity=include_identity,
                                                scale=scale
                                                )

    direction_encoder = BarfPositionalEncoding(levels=fourier_levels_dir,
                                                    alpha_start=fourier_levels_dir,
                                                    alpha_increase_start_epoch=1.28,
                                                    alpha_increase_end_epoch=6.4,
                                                    include_identity=include_identity,
                                                    scale=scale
                                                    )

    model_radiance = NerfModel(
        n_hidden=n_hidden,
        hidden_dim=hidden_dim,
        delayed_direction=delayed_direction,
        delayed_density=delayed_density,
        n_segments=n_segments,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        learning_rate_start=learning_rate_start,
        learning_rate_stop=learning_rate_stop,
        learning_rate_decay_end=learning_rate_decay_end,
    )

    if samples_per_ray_proposal > 0:
        model_proposal = NerfModel(
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            delayed_direction=delayed_direction,
            delayed_density=delayed_density,
            n_segments=n_segments,
            position_encoder=position_encoder,
            direction_encoder=direction_encoder,
            learning_rate_start=learning_rate_start,
            learning_rate_stop=learning_rate_stop,
            learning_rate_decay_end=learning_rate_decay_end,
        )
    else:
        model_proposal=None


    # # Set up model
    model = NerfInterpolationOurs(
        near_sphere_normalized=near_sphere_normalized,
        far_sphere_normalized=far_sphere_normalized,
        samples_per_ray_radiance=samples_per_ray_radiance,
        samples_per_ray_proposal=samples_per_ray_proposal,
        model_radiance=model_radiance,
        model_proposal=model_proposal,
        uniform_sampling_strategy=uniform_sampling_strategy,
        uniform_sampling_offset_size=uniform_sampling_offset_size,
        integration_strategy=integration_strategy,
    )

    return model


def mip_barf_builder(
    fourier_levels_pos = 10,
    fourier_levels_dir = 4,
    include_identity = True,
    samples_per_ray_radiance = 256,
    samples_per_ray_proposal = 64,
    near_sphere_normalized = 2.,
    far_sphere_normalized = 8.,
    scale = 1.,
    n_hidden=4,
    hidden_dim=256,
    delayed_direction=True,
    delayed_density=False,
    n_segments=2,
    learning_rate_start=5e-4,
    learning_rate_stop=1e-4,
    learning_rate_decay_end=200000,
    start_gaussian_sigma = 0.,
    camera_learning_rate_start=1e-3,
    camera_learning_rate_stop=1e-5,
    camera_learning_rate_decay_end=200000,
    max_gaussian_sigma=None,
    n_training_images=None,
    distribute_variance=True,
): 
    
    if n_training_images is None: raise ValueError("n_training_images must be specified")

    position_encoder = IntegratedFourierFeatures(levels=fourier_levels_pos,
                                                scale=scale,
                                                include_identity=include_identity,
                                                distribute_variance=distribute_variance
                                                )

    direction_encoder = BarfPositionalEncoding(levels=fourier_levels_dir,
                                                    alpha_start=fourier_levels_dir,
                                                    alpha_increase_start_epoch=1.28,
                                                    alpha_increase_end_epoch=6.4,
                                                    include_identity=include_identity,
                                                    scale=scale
                                                    )

    model_radiance = NerfModel(
        n_hidden=n_hidden,
        hidden_dim=hidden_dim,
        delayed_direction=delayed_direction,
        delayed_density=delayed_density,
        n_segments=n_segments,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        learning_rate_start=learning_rate_start,
        learning_rate_stop=learning_rate_stop,
        learning_rate_decay_end=learning_rate_decay_end,
    )

    if samples_per_ray_proposal > 0:
        model_proposal = NerfModel(
            n_hidden=n_hidden,
            hidden_dim=hidden_dim,
            delayed_direction=delayed_direction,
            delayed_density=delayed_density,
            n_segments=n_segments,
            position_encoder=position_encoder,
            direction_encoder=direction_encoder,
            learning_rate_start=learning_rate_start,
            learning_rate_stop=learning_rate_stop,
            learning_rate_decay_end=learning_rate_decay_end,
        )
    else:
        model_proposal=None


    model = MipBarf(
        n_training_images=n_training_images,
        camera_learning_rate_start=camera_learning_rate_start,
        camera_learning_rate_stop=camera_learning_rate_stop,
        camera_learning_rate_decay_end=camera_learning_rate_decay_end,
        max_gaussian_sigma=max_gaussian_sigma,
        start_gaussian_sigma=start_gaussian_sigma,
        near_sphere_normalized=near_sphere_normalized,
        far_sphere_normalized=far_sphere_normalized,
        samples_per_ray_radiance=samples_per_ray_radiance,
        samples_per_ray_proposal=samples_per_ray_proposal,
        model_radiance=model_radiance,
        uniform_sampling_strategy = "stratified_uniform",
        uniform_sampling_offset_size=-1.,
        integration_strategy="middle",
    )

    return model



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--uniform_sampling_strategy", type=str, default="stratified_uniform")
    # parser.add_argument("--integration_strategy", type=str, default="middle")
    # parser.add_argument("--uniform_sampling_offset_size", type=float, default=-1.)
    # parser.add_argument("--use_proposal", action=argparse.BooleanOptionalAction, default=True)
    # args = parser.parse_args()
    # # print(args)
    # print(f"testing integration/sample_strats - proposal={args.use_proposal}, {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}")

    # quit()
    # exit()

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="testing mip nerf",
        # name=f"testing integration/sample_strats - proposal={args.use_proposal}, {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}",
    )


    # Set up data module
    BATCH_SIZE = 1024*2
    NUM_WORKERS = 3
    # IMAGE_SIZE = 200
    IMAGE_SIZE = 400
    # IMAGE_SIZE = 40
    SIGMAS_FOR_BLUR = [16, 8, 4, 2, 1, 0.5, 0.]
    # SIGMAS_FOR_BLUR = [4, 3, 2, 1, 0.]

    
    dm = ImagePoseDataModule(
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
        space_transform_scale=1.,
        space_transform_translate=th.Tensor([0,0,0]),
        scene_path="../data/lego",
        verbose=True,
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        gaussian_blur_sigmas = SIGMAS_FOR_BLUR,
        rotation_noise_sigma = 0,#.15,
        translation_noise_sigma = 0,#.15,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
    )

    dm.setup("fit")


    # Set up trainer
    th.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=100,
        # precision="32-mixed",
        logger=wandb_logger,
        callbacks=[
            LogEpochFraction(
                wandb_logger=wandb_logger,
                metric_name="epoch_fraction",
            ),
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                logging_start=0.002,
                # delay_start=1/2,
                # delay_start=1/16,
                delay_start=1/24,
                delay_end=1.,
                delay_taper=5.0,
                train_image_names=["r_1", "r_23"],
                validation_image_names=["r_2", "r_84"],
                reconstruction_batch_size=BATCH_SIZE,
                reconstruction_num_workers=NUM_WORKERS,
                metric_name_val="val_img",
                metric_name_train="train_img",
            ),
            # LogCameraExtrinsics(
            #     wandb_logger=wandb_logger,
            #     logging_start=0.000,
            #     delay_start=1/200,
            #     delay_end=1/16,
            #     delay_taper=4.0,
            #     batch_size=BATCH_SIZE,
            #     num_workers=NUM_WORKERS,
            #     ray_direction_length=1/10,
            #     metric_name="train_point",
            # ),
            LearningRateMonitor(
                logging_interval="step"
            ),
            ModelCheckpoint(
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.3e}',
                every_n_epochs=1,
                save_top_k=-1,
            ),
            # dm.get_dataset_blur_scheduler_callback(
            #     epoch_fraction_period=0.02,
            #     dataset_name="train"
            # ),
            # dm.get_dataset_blur_scheduler_callback(
            #     epoch_fraction_period=0.02,
            #     dataset_name="val"
            # )
        ]
    )


    position_encoder = IntegratedFourierFeatures(levels=10, 
                                                 scale=1.,
                                                 include_identity=True,
                                                 distribute_variance=True)
    # position_encoder = BarfPositionalEncoding(levels=10,
    #                                             # alpha_start=0,
    #                                             alpha_start=10,
    #                                             alpha_increase_start_epoch=1.28,
    #                                             alpha_increase_end_epoch=6.4,
    #                                             include_identity=True,
    #                                             scale=1.
    #                                             )
    direction_encoder = BarfPositionalEncoding(levels=4,
                                                    alpha_start=4,
                                                    # alpha_start=0,
                                                    alpha_increase_start_epoch=1.28,
                                                    alpha_increase_end_epoch=6.4,
                                                    include_identity=True,
                                                    scale=1.
                                                    )

    model_radiance_and_proposal = NerfModel(
        n_hidden=4,
        hidden_dim=256,
        delayed_direction=True,
        delayed_density=False,
        n_segments=2,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        learning_rate_start=5e-4,
        learning_rate_stop=1e-4,
        learning_rate_decay_end=200000,
    )
    # if args.use_proposal:
    #     model_proposal = NerfModel(
    #         n_hidden=4,
    #         hidden_dim=256,
    #         delayed_direction=True,
    #         delayed_density=False,
    #         n_segments=2,
    #         position_encoder=position_encoder,
    #         direction_encoder=direction_encoder,
    #         learning_rate_start=5e-4,
    #         learning_rate_stop=1e-4,
    #         learning_rate_decay_end=200000,
    #     )
    #     samples_per_ray_proposal=32,

    # else:
    #     model_proposal=None
    #     samples_per_ray_proposal=0,

    # model = MipNeRF(
    #     near_sphere_normalized=2.,
    #     far_sphere_normalized=8.,
    #     samples_per_ray_radiance=256,
    #     samples_per_ray_proposal=64,
    #     uniform_sampling_strategy="stratified_uniform",
    #     uniform_sampling_offset_size=-1.,
    #     integration_strategy="middle",
    #     # samples_per_ray_radiance=256,
    #     # samples_per_ray_proposal=64,
    #     # uniform_sampling_strategy=args.uniform_sampling_strategy,
    #     # uniform_sampling_offset_size=args.uniform_sampling_offset_size,
    #     # integration_strategy=args.integration_strategy,
    #     model_radiance=model_radiance_and_proposal,
    #     model_proposal=model_radiance_and_proposal,
    # )

    # # Set up model
    model = MipBarf(
        n_training_images=len(dm.dataset_train.images),
        camera_learning_rate_start=0,
        camera_learning_rate_stop=0,
        camera_learning_rate_decay_end=200000,
        # camera_learning_rate_start=1e-3,
        # camera_learning_rate_stop=1e-5,
        # camera_learning_rate_decay_end=200000,
        max_gaussian_sigma=max(SIGMAS_FOR_BLUR),
        start_gaussian_sigma=15,
        near_sphere_normalized=2.,
        far_sphere_normalized=8.,
        samples_per_ray_radiance=128,
        samples_per_ray_proposal=64,
        model_radiance=model_radiance_and_proposal,
        model_proposal=model_radiance_and_proposal,
    )
    # # # Set up model
    # model = BarfModel(
    #     n_training_images=len(dm.dataset_train.images),
    #     camera_learning_rate_start=1e-3,
    #     camera_learning_rate_stop=1e-5,
    #     camera_learning_rate_decay_end=200000,
    #     max_gaussian_sigma=max(SIGMAS_FOR_BLUR),
    #     near_sphere_normalized=2.,
    #     far_sphere_normalized=8.,
    #     samples_per_ray_radiance=128,
    #     # samples_per_ray_proposal=64,
    #     model_radiance=model_radiance,
    #     # model_proposal=model_proposal,
    # )

    model = NerfInterpolationOurs(
        
    )


    # wandb_logger.watch(model, log="all")

    # Start training, resume from checkpoint
    trainer.fit(model, dm)#, ckpt_path="/work3/s204111/nerf-experiments/barf/nerf-experiments/vq9nm9vt/checkpoints/ckpt_epoch=epoch=03-val_loss=val_loss=0.00.ckpt")