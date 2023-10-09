from math import log2

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from epoch_fraction_logger import LogEpochFraction
from model import Garf


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="garf",
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
                delay_start=1/200,
                delay_end=1/16,
                delay_taper=4.0,
                validation_image_names=["r_2", "r_84"],
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
    LEARNING_RATE_START = 5e-3
    LEARNING_RATE_STOP = 5e-3

    model = Garf(
        near_sphere_normalized=2,
        far_sphere_normalized=7,
        samples_per_ray_coarse=64,
        samples_per_ray_fine=192,
        gaussian_init_min=0.,
        gaussian_init_max=1.,
        learning_rate=LEARNING_RATE_START,
        learning_rate_decay=2**(log2(LEARNING_RATE_STOP/LEARNING_RATE_START) / trainer.max_epochs), # type: ignore
        weight_decay=0
    )


    # Start training, resume from checkpoint
    trainer.fit(model, dm)
