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

# High priority 
# TODO: Script that generates runs where the plots are used for the paper 
# TODO: Bug hunting
# TODO: Write paper 

# Low priority
# TODO: Fix image logger - see that file
# TODO: Remove all the runs in WANDB (davids are gone)
# TODO: Use decay of learning rate - Check if it works in the bottom of model_camera_calibration - for camera extrinsics go as default from 1e-3 to 1e-5. For normal nerf from 5e-4 to 1e-4 
# TODO: dataset.py: Rewrite datamodule to save the transformed images as (blurred images e.g)
#       such that it can be easily read when instantiating a dataset with sigmas that
#       have already been calculated once. This will save a lot of time during startup of a run.
#       May not be a good idea tho, as it would require a lot of memory to store all the images??
# TODO: We should try to run an experiment where we simulate BARF,
#       but without our space transformation to the unit sphere.
# TODO: Run experiment where we use the so3_to_SO3 from BARF - see Lie_barf.py
#       

# Converter that takes iterations to epochs to adjust alpha
def convert_iterations_to_epochs(iterations: int, batch_size: int, dataset_size_samples: int) -> float:
    return iterations * batch_size / dataset_size_samples


if __name__ == "__main__":

    # Set seeds
    pl.seed_everything(1337, workers=True)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="mip blur"
    )


    # Set up data module
    BATCH_SIZE = 1024*1
    NUM_WORKERS = 3
    # IMAGE_SIZE = 200
    IMAGE_SIZE = 400
    # IMAGE_SIZE = 40
    # SIGMAS_FOR_BLUR = [0.]
    SIGMAS_FOR_BLUR = [16, 0.]
    # SIGMAS_FOR_BLUR = [16, 8, 4, 2, 1, 0.5, 0.]
    # SIGMAS_FOR_BLUR = [4, 3, 2, 1, 0.]

    
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

    # dm.setup("fit")


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
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.3e}',
                every_n_epochs=1,
                save_top_k=-1,
            ),
        ]
    )


    # model = nerf_interpolation_builder()

    model = mip_barf_builder(
        start_gaussian_sigma=15,
        camera_learning_rate_start=0,
        camera_learning_rate_stop=0,
        camera_learning_rate_decay_end=-1,
        distribute_variance=True,
        n_training_images=dm.n_training_images,

    )

    trainer.fit(model, dm)#, ckpt_path="/work3/s204111/nerf-experiments/barf/nerf-experiments/vq9nm9vt/checkpoints/ckpt_epoch=epoch=03-val_loss=val_loss=0.00.ckpt")