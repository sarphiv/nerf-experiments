from math import log2
import argparse
import os

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from epoch_fraction_logger import LogEpochFraction
from model_interpolation import NerfInterpolation
from mip_model import MipNerf


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_fourier', type=bool, default=True, help='Whether to use Fourier features or not')
    parser.add_argument('--use_proposal', type=bool, default=True, help='Whether to have a proposal network or not')
    parser.add_argument('--delayed_direction', type=bool, default=True, help='When the directional input is feed to the network')
    parser.add_argument('--delayed_density', type=bool, default=True, help='When the network outputs the density')
    parser.add_argument('--n_segments', type=int, default=2, help='Number of times the positional data is feed to the network')
    parser.add_argument('--n_hidden', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--mip_distribute_variance', type=bool, default=True, help='Whether to distribute the variance in the MIP model or not')
    parser.add_argument('--experiment_name', type=str, default=f"Unnamed experiment at '{os.path.basename(os.path.dirname(__file__))}'")
    parser.add_argument('--use_seperate_coarse_fine', type=bool, default=False, help='Whether to use seperate coarse and fine models or one model for both')
    args = parser.parse_args()

    print("Using arguments:")
    print(args)

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name=args.experiment_name
    )


    # Set up data module
    BATCH_SIZE = 1024*2
    
    dm = ImagePoseDataModule(
        image_width=800,
        image_height=800,
        scene_path="../data/lego",
        validation_fraction=0.05,
        validation_fraction_shuffle=1234,
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=True,
    )

    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=100,
        precision="32",
        logger=wandb_logger,
        callbacks=[
            LogEpochFraction(
                wandb_logger=wandb_logger,
                metric_name="epoch_fraction",
            ),
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                logging_start=0.002,
                delay_start=0.05,
                delay_end=1/4.,
                delay_taper=2.0,
                validation_image_names=["r_2"],#, "r_84"],
                reconstruction_batch_size=BATCH_SIZE,
                reconstruction_num_workers=4,
                metric_name="val_img",
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            ),
            ModelCheckpoint(
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
                every_n_epochs=2,
                save_top_k=-1,
            )
        ]
    )


    # Set up model
    model = MipNerf(
        near_sphere_normalized=1/10,
        far_sphere_normalized=1/3,
        samples_per_ray=64 + 192,
        n_hidden=args.n_hidden,
        fourier=(args.use_fourier, 10, 4),
        proposal=(args.use_proposal, 64),
        n_segments=args.n_segments,
        learning_rate=5e-4,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        weight_decay=0,
        distribute_variance=args.mip_distribute_variance,
        seperate_coarse_fine=args.use_seperate_coarse_fine,
    )


    # Start training, resume from checkpoint
    trainer.fit(model, dm)
