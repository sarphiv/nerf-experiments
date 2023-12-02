from math import log2
from dataclasses import dataclass

import tyro
import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from point_logger import LogCameraExtrinsics
from epoch_fraction_logger import LogEpochFraction
from model_builders import nerf_interpolation_builder, mip_barf_builder
from model_camera_calibration import CameraCalibrationModel
from model_interpolation_architecture import NerfModel
from positional_encodings import BarfPositionalEncoding, IntegratedFourierFeatures, IntegratedBarfFourierFeatures, FourierFeatures
from model_interpolation import NerfInterpolation, uniform_sampling_strategies, integration_strategies
from model_barf import BarfModel
from model_mip import MipNeRF, MipBarf

@dataclass(frozen=True)
class Args:
    camera_origin_noise_sigma: float = 0.15
    camera_rotation_noise_sigma: float = 0.15 
    start_blur_sigma: float = 200. 
    start_pixel_width_sigma: float = 200
    max_blur_sigma: float = 200 
    n_blur_sigmas: int = 10
    seed: int = 134534 
    optimize_camera: bool = True


if __name__ == "__main__":

    args = tyro.cli(Args)
    print(args)
    # print(f"mip that test barf runs")
    # Set up data module
    BATCH_SIZE = 1024
    NUM_WORKERS = 3
    IMAGE_SIZE = 400

    if args.max_blur_sigma <= 0.25:
        SIGMAS_FOR_BLUR = [0.0, 0.0]
    elif args.n_blur_sigmas <= 2:
        SIGMAS_FOR_BLUR = [args.max_blur_sigma, 0.0]
    else:
        SIGMAS_FOR_BLUR = (2**th.flip(th.linspace(-1, log2(args.max_blur_sigma), args.n_blur_sigmas - 1), dims=(0,))).round(decimals=2).tolist() + [0.0] 


    print(SIGMAS_FOR_BLUR)
    LR_DECAY_END_STEP = 200000
    SIGMA_SCHEDULE_DECAY_START_STEP = 2000 #TODO 20000
    SIGMA_SCHEDULE_DECAY_END_STEP = 100000



    # BATCH_SIZE = 256 # TODO: remove 

    # quit()
    # exit()

    # Set seeds
    pl.seed_everything(args.seed, workers=True)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name=f"bipBARF noise={args.camera_origin_noise_sigma} blur={args.start_blur_sigma} pixel_width={args.start_pixel_width_sigma} sigmas_for_blur={SIGMAS_FOR_BLUR}"
        # name=f"testing integration/sample_strats - proposal={args.use_proposal}, {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}",
    )

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
        rotation_noise_sigma = args.camera_rotation_noise_sigma,
        translation_noise_sigma = args.camera_origin_noise_sigma,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        camera_noise_seed=args.seed
    )


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
            LogCameraExtrinsics(
                wandb_logger=wandb_logger,
                logging_start=0.000,
                delay_start=1/200,
                delay_end=1/16,
                delay_taper=4.0,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                ray_direction_length=1/10,
                metric_name="train_point",
            ),
            LearningRateMonitor(
                logging_interval="step"
            ),
            ModelCheckpoint(
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.3e}',
                every_n_epochs=1,
                save_top_k=-1,
            ),
        ]
    )


    position_encoder = IntegratedFourierFeatures(
        levels=10,
        include_identity=True,
        scale=1.,
        distribute_variance=False,
    )

    direction_encoder = BarfPositionalEncoding(0, 1, 0, 1, True)

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
        learning_rate_decay_end=LR_DECAY_END_STEP
    )


    model = MipBarf(
        n_training_images=dm.n_training_images,
        camera_learning_rate_start=1e-3 if args.optimize_camera else 0.,
        camera_learning_rate_stop=1e-5 if args.optimize_camera else 0.,
        camera_learning_rate_decay_end=LR_DECAY_END_STEP,
        near_sphere_normalized=2,
        far_sphere_normalized=8,
        samples_per_ray_radiance=126,# 256,
        samples_per_ray_proposal=0,# 64,
        model_radiance=model_radiance,
        uniform_sampling_strategy = "equidistant",
        uniform_sampling_offset_size=-1.,
        sigma_decay_start_step=SIGMA_SCHEDULE_DECAY_START_STEP,
        sigma_decay_end_step=SIGMA_SCHEDULE_DECAY_END_STEP,
        start_blur_sigma=args.start_blur_sigma,
        start_pixel_width_sigma=args.start_pixel_width_sigma,
    )


    trainer.fit(model, dm)#, ckpt_path="/work3/s204111/nerf-experiments/barf/nerf-experiments/vq9nm9vt/checkpoints/ckpt_epoch=epoch=03-val_loss=val_loss=0.00.ckpt")