import torch as th
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # type: ignore
from pytorch_lightning.callbacks import LearningRateMonitor

from model import Nerf2d
from data_loader import SingleImageDataModule
from image_logger import Log2dImageReconstruction





if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger"
    )


    # Set up data module
    dm = SingleImageDataModule(
        # image_path="data/morten-lena.png",
        image_path="data/banana.jpg",
        pixel_shuffle_seed=1337,
        pixel_split_sizes=(0.9, 0.05, 0.05),
        batch_size=256,
        num_workers=4,
        shuffle=True
    )
    dm.setup(None)


    # Set up model
    th.set_float32_matmul_precision("medium")

    model = Nerf2d(
        width=dm.image_width, 
        height=dm.image_height, 
        fourier_levels=10,
        learning_rate=1e-4,
        learning_rate_decay=0.5,
        learning_rate_decay_patience=80,
        weight_decay=0
    )


    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=512,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                epoch_period=20,
                width=dm.image_width, 
                height=dm.image_height
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            )
        ]
    )


    # Start training
    trainer.fit(model, dm)

