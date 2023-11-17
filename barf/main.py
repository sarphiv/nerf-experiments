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
from model_interpolation_architecture import BarfPositionalEncoding




if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotation_noise", type=float, default=0.0)
    parser.add_argument("--translation_noise", type=float, default=0.0)
    parser.add_argument('--use_fourier', type=bool, default=True, help='Whether to use Fourier features or not')
    parser.add_argument('--use_proposal', type=bool, default=True, help='Whether to have a proposal network or not')
    parser.add_argument('--delayed_direction', type=bool, default=True, help='When the directional input is feed to the network')
    parser.add_argument('--delayed_density', type=bool, default=True, help='When the network outputs the density')
    parser.add_argument('--n_segments', type=int, default=2, help='Number of times the positional data is fed to the network')
    parser.add_argument('--n_hidden', type=int, default=4, help='Number of hidden layers')
    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="testing new dataset"
    )


    # Set up data module
    BATCH_SIZE = 1024*2
    NUM_WORKERS = 1
    
    dm = ImagePoseDataModule(
        image_width=400,
        image_height=400,
        scene_path="../data/lego",
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        gaussian_blur_sigmas =[3.0, 1.92, 1.08, 0.48, 0.0],
        rotation_noise_sigma = 0.0,
        translation_noise_sigma = 0.0,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )
    
    # dm = ImagePoseDataModule(
    #     image_width=80,
    #     image_height=80,
    #     scene_path="../data/lego",
    #     space_transform_scale=None,
    #     space_transform_translate=None,
    #     rotation_noise_sigma=float(args.rotation_noise),
    #     translation_noise_sigma=float(args.translation_noise),
    #     camera_noise_seed=13571113,
    #     gaussian_blur_kernel_size=81,
    #     gaussian_blur_relative_sigma_start=0.,
    #     gaussian_blur_relative_sigma_decay=0.99,
    #     validation_fraction=0.05,
    #     validation_fraction_shuffle=1234,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     shuffle=True,
    #     pin_memory=True
    # )


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
                delay_start=1/7,
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
                logging_start=0.002,
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
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
                every_n_epochs=2,
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


    # Set up model
    model = CameraCalibrationModel(
        n_training_images=len(dm.dataset_train.images),
        # camera_learning_rate=5e-4,
        camera_learning_rate=0.,
        camera_learning_rate_stop_epoch=8,
        camera_learning_rate_decay=0.999,
        camera_learning_rate_period=0.02,
        camera_weight_decay=0.0,
        near_sphere_normalized=1/10,
        far_sphere_normalized=1/3,
        samples_per_ray=64 + 192,
        n_hidden=args.n_hidden,
        hidden_dim=256,
        position_encoder = BarfPositionalEncoding(levels=10,
                                                  alpha_start=0,
                                                #   alpha_increase_start_epoch=0.,
                                                  alpha_increase_start_epoch=1.28,
                                                  alpha_increase_end_epoch=6.4,
                                                  include_identity=True),
        direction_encoder = BarfPositionalEncoding(levels=4,
                                                     alpha_start=4,
                                                     alpha_increase_start_epoch=1.28,
                                                     alpha_increase_end_epoch=6.4,
                                                     include_identity=True),
        max_gaussian_sigma=2.0,
        proposal=(args.use_proposal, 64),
        delayed_direction=args.delayed_direction,
        delayed_density=args.delayed_density,
        n_segments=args.n_segments,
        learning_rate=5e-4,
        learning_rate_stop_epoch = 100,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        learning_rate_period = 1.0,
        weight_decay=0
    )


    wandb_logger.watch(model, log="all")

    # Start training, resume from checkpoint
    trainer.fit(model, dm)