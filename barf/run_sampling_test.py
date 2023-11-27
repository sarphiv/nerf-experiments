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
from model_builders import nerf_interpolation_builder, mip_barf_builder

argparse.Action

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform_sampling_strategy", type=str, default="stratified_uniform")
    parser.add_argument("--integration_strategy", type=str, default="middle")
    parser.add_argument("--uniform_sampling_offset_size", type=float, default=-1.)
    args = parser.parse_args()
    # print(args)
    print(f"testing integration/sample_strats - {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}")

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name=f"testing integration/sample_strats - {args.uniform_sampling_strategy[:5]}, {args.integration_strategy[:4]}, {args.uniform_sampling_offset_size}",
    )


    # Set up data module
    BATCH_SIZE = 1024*2
    NUM_WORKERS = 3
    IMAGE_SIZE = 400
    # IMAGE_SIZE = 40
    SIGMAS_FOR_BLUR = [0.]

    
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


    model = nerf_interpolation_builder(
        uniform_sampling_strategy=args.uniform_sampling_strategy,
        uniform_sampling_offset_size=args.uniform_sampling_offset_size,
        integration_strategy=args.integration_strategy,
    )

    trainer.fit(model, dm)#, ckpt_path="/work3/s204111/nerf-experiments/barf/nerf-experiments/vq9nm9vt/checkpoints/ckpt_epoch=epoch=03-val_loss=val_loss=0.00.ckpt")