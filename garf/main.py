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



if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="garf-merge-test",
    )


    # Set up data module
    BATCH_SIZE = 1024
    NUM_WORKERS = 8

    dm = ImagePoseDataModule(
        image_width=800,
        image_height=800,
        scene_path="../data/lego",
        space_transform_scale=1,
        space_transform_translate=None,
        gaussian_blur_sigmas=[ 0.0 ],
        rotation_noise_sigma=0.15,
        translation_noise_sigma=0.15,
        camera_noise_seed=13571113,
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    dm.setup("fit")


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=20,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            LogEpochFraction(
                wandb_logger=wandb_logger,
                metric_name="epoch_fraction",
            ),
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                logging_start=0.002,
                delay_start=1/16,
                delay_end=1/4,
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
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
                every_n_epochs=1,
                save_top_k=-1,
            )
        ]
    )


    # Set up model
    # NOTE: Period is in epoch fractions
    CAMERA_LEARNING_RATE_START = 1e-4
    CAMERA_LEARNING_RATE_STOP = 1e-5
    CAMERA_LEARNING_RATE_STOP_EPOCH = 3
    CAMERA_LEARNING_RATE_PERIOD = 0.01
    CAMERA_LEARNING_RATE_DECAY: float = 2**(log2(CAMERA_LEARNING_RATE_STOP/CAMERA_LEARNING_RATE_START) * CAMERA_LEARNING_RATE_PERIOD/CAMERA_LEARNING_RATE_STOP_EPOCH) # type: ignore
    
    PROPOSAL_LEARNING_RATE_START = 5e-4
    PROPOSAL_LEARNING_RATE_STOP = 5e-6
    PROPOSAL_LEARNING_RATE_STOP_EPOCH = 8
    PROPOSAL_LEARNING_RATE_PERIOD = 0.01
    PROPOSAL_LEARNING_RATE_DECAY: float = 2**(log2(PROPOSAL_LEARNING_RATE_STOP/PROPOSAL_LEARNING_RATE_START) * PROPOSAL_LEARNING_RATE_PERIOD/PROPOSAL_LEARNING_RATE_STOP_EPOCH) # type: ignore

    RADIANCE_LEARNING_RATE_START = 2e-4
    RADIANCE_LEARNING_RATE_STOP = 8e-6
    RADIANCE_LEARNING_RATE_STOP_EPOCH = 8
    RADIANCE_LEARNING_RATE_PERIOD = 0.01
    RADIANCE_LEARNING_RATE_DECAY: float = 2**(log2(RADIANCE_LEARNING_RATE_STOP/RADIANCE_LEARNING_RATE_START) * RADIANCE_LEARNING_RATE_PERIOD/RADIANCE_LEARNING_RATE_STOP_EPOCH) # type: ignore

    model = CameraCalibrationModel(
        n_training_images=len(dm.dataset_train.images),
        camera_learning_rate=CAMERA_LEARNING_RATE_START,
        camera_learning_rate_stop_epoch=CAMERA_LEARNING_RATE_STOP_EPOCH,
        camera_learning_rate_decay=CAMERA_LEARNING_RATE_DECAY,
        camera_learning_rate_period=CAMERA_LEARNING_RATE_PERIOD,
        camera_weight_decay=0.0,

        near_plane=2,
        far_plane=7,
        proposal_samples_per_ray=64,
        radiance_samples_per_ray=192,
        gaussian_init_min=1/2.,
        gaussian_init_max=16,
        gaussian_learning_rate_factor=128.,
        proposal_learning_rate=PROPOSAL_LEARNING_RATE_START,
        proposal_learning_rate_stop_epoch=PROPOSAL_LEARNING_RATE_STOP_EPOCH,
        proposal_learning_rate_decay=PROPOSAL_LEARNING_RATE_DECAY,
        proposal_learning_rate_period=PROPOSAL_LEARNING_RATE_PERIOD,
        proposal_weight_decay=1e-8,
        radiance_learning_rate=RADIANCE_LEARNING_RATE_START,
        radiance_learning_rate_stop_epoch=PROPOSAL_LEARNING_RATE_STOP_EPOCH,
        radiance_learning_rate_decay=RADIANCE_LEARNING_RATE_DECAY,
        radiance_learning_rate_period=RADIANCE_LEARNING_RATE_PERIOD,
        radiance_weight_decay=1e-9,
    )

    # Log model gradients and parameters
    # wandb_logger.watch(model, log="all")

    # Start training
    trainer.fit(model, dm)
