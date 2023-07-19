from math import log2

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
from pytorch_lightning.loggers import WandbLogger, CSVLogger  # type: ignore



from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from model import NaiveINGP, FourierFeatures, INGPEncoding
from plotter import Plotter

from pytorch_lightning.profilers import AdvancedProfiler
import os


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)

    EXPERIMENTS_PATH = "experiments"
    EXPERIMENT_NAME = "pretty_please_work"
    # EXPERIMENT_NAME = "test_nerf_original"
    


    path = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME)

    os.makedirs(path, exist_ok=True)

    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger"
    )

    # Set up data module
    dm = ImagePoseDataModule(
        image_width=50,
        image_height=50,
        scene_path="../data/lego",
        validation_fraction=0.05,
        validation_fraction_shuffle=1234,
        batch_size=1024*5,
        num_workers=4,
        shuffle=True,
    )


    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(path,'checkpoints'),
        filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2e}',
        every_n_epochs=1,
        save_top_k=3,
        monitor='val_loss',
        )

    os.makedirs(os.path.join(path, "output_images"), exist_ok=True)

    # Set up trainer
    th.set_float32_matmul_precision("medium")

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=256,
        # precision="bf16",
        precision="16-mixed",
        logger=wandb_logger,
        # logger=csvlogger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                epoch_period=1,
                validation_image_name="r_2",
                batch_size=1024*5,
                num_workers=4,
                path=os.path.join(path, "output_images"),
                plot_period=None,
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            ),
            checkpoint_callback_time,
            Plotter(os.path.join(path, "loss_plot.png")),
        ]
        , profiler=profiler
    )


    direction_encoder = FourierFeatures(levels = 4)

    position_encoder = FourierFeatures(levels = 4)

    # position_encoder = INGPEncoding(
    #     resolution_max=1600,
    #     resolution_min=16,
    #     table_size=2**16,
    #     n_features=2,
    #     n_levels=16
    # )

    model = NaiveINGP(
        near_sphere_normalized=2,
        far_sphere_normalized=7,
        samples_per_ray_coarse=64,
        samples_per_ray_fine=192,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        n_hidden=8,
        hidden_dim=256,
        learning_rate=1e-5,
        learning_rate_decay=1,#2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        weight_decay=0,
    )


    # Start training, resume from checkpoint
    trainer.fit(model, dm)
