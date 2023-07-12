from math import log2

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from model import NerfOriginal, NaiveINGP

from pytorch_lightning.profilers import AdvancedProfiler
import os


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)

    EXPERIMENTS_PATH = "experiments"
    EXPERIMENT_NAME = "test_nerf_naive_ingp_2"
    # EXPERIMENT_NAME = "test_nerf_original"
    


    path = os.path.join(EXPERIMENTS_PATH, EXPERIMENT_NAME)

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
        batch_size=1024*4,
        num_workers=4,
        shuffle=True,
    )


    checkpoint_callback_time = ModelCheckpoint(
        dirpath=os.path.join(path,'checkpoints'),
        filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
        every_n_epochs=1,
        save_top_k=-1,
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
        # logger=wandb_logger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                epoch_period=1,
                validation_image_name="r_2",
                batch_size=1024*4,
                num_workers=4,
                path=os.path.join(path, "output_images"),
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            ),
            checkpoint_callback_time
        ]
        , profiler=profiler
    )


    # Set up model
    # model = NerfOriginal(
    #     near_sphere_normalized=2,
    #     far_sphere_normalized=7,
    #     samples_per_ray_coarse=64,
    #     samples_per_ray_fine=192,
    #     fourier_levels_pos=10,
    #     fourier_levels_dir=4,
    #     learning_rate=5e-4,
    #     learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
    #     weight_decay=0,
    #     use_residual_color=True,
    # )

    model = NaiveINGP(
        near_sphere_normalized=2,
        far_sphere_normalized=7,
        samples_per_ray_coarse=64,
        samples_per_ray_fine=192,
        fourier_levels_dir=4,
        resolution_max=50,
        resolution_min=4,
        table_size=2**10,
        learning_rate=1e-2,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        weight_decay=0,
        use_residual_color=True,
    )


    # Start training, resume from checkpoint
    trainer.fit(model, dm)
