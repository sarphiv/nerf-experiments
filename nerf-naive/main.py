from math import log2

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from model import NerfNaive


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="nerf-naive-comparison"
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
        max_epochs=256,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                batch_period=100,
                epoch_period=1,
                validation_image_name="r_2",
                batch_size=BATCH_SIZE,
                num_workers=4
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
    model = NerfNaive(
        near_sphere_normalized=1/10,
        far_sphere_normalized=1/3,
        samples_per_ray=64 + 192,
        learning_rate=5e-4,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        weight_decay=0
    )


    # Start training, resume from checkpoint
    trainer.fit(model, dm)
