import torch as th
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger # type: ignore
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os

from model import Gigapixel, INGPEncoding
from data_loader import SingleImageDataModule
from image_logger import Log2dImageReconstruction



if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)

    EXPERIMENTS_PATH = "experiments"
    EXPERIMENT_NAME = "test-2d-ingp-v100-morten-lena"

    path = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger"
    )

    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(path,'checkpoints'),
        # use sciencific notation for val loss
        filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2e}',
        every_n_epochs=1,
        save_top_k=-1,
        )

    os.makedirs(os.path.join(path, "output_images"), exist_ok=True)

    # Set up data module
    dm = SingleImageDataModule(
        # image_path="data/morten-lena.png",
        image_path="../data/Lena_2048.png",
        pixel_shuffle_seed=1337,
        pixel_split_sizes=(0.9, 0.05, 0.05),
        batch_size=2560,
        num_workers=4,
        shuffle=True
    )
    dm.setup(None)


    # Set up model
    # th.set_float32_matmul_precision("medium")

    position_encoder = INGPEncoding(resolution_max=2048,
                                    resolution_min=16,
                                    table_size=2**16,
                                    n_features=2,
                                    n_levels=16)

    model = Gigapixel(4, 256, position_encoder)

    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=512,
        # precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger,
                epoch_period=1,
                width=dm.image_width,
                height=dm.image_height,
                path=os.path.join(path, "output_images")
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            ),
            checkpoint_callback_time
        ]
    )


    # Start training
    trainer.fit(model, dm)

