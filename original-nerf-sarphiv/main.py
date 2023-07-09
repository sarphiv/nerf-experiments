from math import log2

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from model import NerfOriginal


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)

    print("Starting training...")
    # check if we have a GPU available
    if th.cuda.is_available():
        device = th.device("cuda")
        print("Using GPU")
    else:
        device = th.device("cpu")
        print("Using CPU")

    # Set up weights and biases logger
    # wandb_logger = WandbLogger(
    #     project="nerf-experiments", 
    #     entity="metrics_logger"
    # )


    # Set up data module
    dm = ImagePoseDataModule(
        image_width=100,
        image_height=100,
        scene_path="../data/lego",
        validation_fraction=0.05,
        validation_fraction_shuffle=1234,

        batch_size=1024*4,
        num_workers=8,
        shuffle=True,
    )


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=256,
        precision="bf16",
        # precision="16-mixed",
        # logger=wandb_logger,
        callbacks=[
            # Log2dImageReconstruction(
                # wandb_logger=wandb_logger,
                # epoch_period=1,
                # validation_image_name="r_2",
                # batch_size=1024*4,
                # num_workers=8
            # ),
            LearningRateMonitor(
                logging_interval="epoch"
            )
        ]
    )


    # Set up model
    model = NerfOriginal(
        near_sphere_normalized=2,
        far_sphere_normalized=7,
        samples_per_ray_coarse=64,
        samples_per_ray_fine=192,
        fourier_levels_pos=10,
        fourier_levels_dir=4,
        learning_rate=5e-4,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        weight_decay=0
    )


    # Start training
    trainer.fit(model, dm)
