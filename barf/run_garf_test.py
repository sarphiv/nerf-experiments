from dataclasses import dataclass

import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger  # type: ignore
import tyro

from data_module import ImagePoseDataModule
from image_logger import Log2dImageReconstruction
from point_logger import LogCameraExtrinsics
from epoch_fraction_logger import LogEpochFraction
from model_garf import GarfModel



@dataclass(frozen=True)
class Args:
    name: str | None = None
    
    camera_origin_noise_sigma: float = 0.15
    camera_rotation_noise_sigma: float = 0.15
    camera_learning_rate_start: float = 2e-3
    camera_learning_rate_stop: float = 5e-5
    camera_learning_rate_decay_end: float = 5.0
    
    gaussian_learning_rate_factor: float = 128.0
    gaussian_init_max: float = 2.0
    gaussian_init_min: float = 0.5
    
    radiance_learning_rate_start: float = 2e-4
    radiance_learning_rate_stop: float = 8e-6
    radiance_learning_rate_decay_end: float = 8.0
    radiance_weight_decay: float = 1e-9
    
    proposal_learning_rate_start: float = 5e-4
    proposal_learning_rate_stop: float = 5e-6
    proposal_learning_rate_decay_end: float = 8.0
    proposal_weight_decay: float = 1e-8
    
    image_size: int = 400
    batch_size: int = 1024
    num_workers: int = 8



if __name__ == "__main__":
    # Retrieve arguments
    args = tyro.cli(Args)


    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name=args.name,
    )


    # Set up data module
    dm = ImagePoseDataModule(
        image_width=args.image_size,
        image_height=args.image_size,
        scene_path="../data/lego",
        space_transform_scale=1,
        space_transform_translate=None,
        gaussian_blur_sigmas=[ 0.0 ],
        rotation_noise_sigma=args.camera_rotation_noise_sigma,
        translation_noise_sigma=args.camera_origin_noise_sigma,
        camera_noise_seed=13571113,
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True
    )


    # Set up trainer
    th.set_float32_matmul_precision("medium")

    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=40,
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
                delay_start=1/16,
                delay_end=1/4,
                delay_taper=5.0,
                train_image_names=["r_1", "r_23"],
                validation_image_names=["r_2", "r_84"],
                reconstruction_batch_size=args.batch_size,
                reconstruction_num_workers=args.num_workers,
                metric_name_val="val_img",
                metric_name_train="train_img",
            ),
            LogCameraExtrinsics(
                wandb_logger=wandb_logger,
                logging_start=0.000,
                delay_start=1/200,
                delay_end=1/16,
                delay_taper=4.0,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                ray_direction_length=1/10,
                metric_name="train_point",
            ),
            LearningRateMonitor(
                logging_interval="step"
            ),
            ModelCheckpoint(
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
                every_n_epochs=1,
                save_top_k=-1,
            )
        ]
    )


    # Set up model
    epoch_fraction_to_steps = lambda epoch: int(epoch * dm.n_training_images * args.image_size**2 / args.batch_size)

    model = GarfModel(
        n_train_images=dm.n_training_images,
        
        camera_learning_rate_start=args.camera_learning_rate_start,
        camera_learning_rate_stop=args.camera_learning_rate_stop,
        camera_learning_rate_decay_end=epoch_fraction_to_steps(args.camera_learning_rate_decay_end),

        near_plane=2,
        far_plane=7,
        proposal_samples_per_ray=64,
        radiance_samples_per_ray=192,

        gaussian_init_min=args.gaussian_init_min,
        gaussian_init_max=args.gaussian_init_max,
        gaussian_learning_rate_factor=args.gaussian_learning_rate_factor,

        proposal_learning_rate_start=args.proposal_learning_rate_start,
        proposal_learning_rate_stop=args.proposal_learning_rate_stop,
        proposal_learning_rate_decay_end=epoch_fraction_to_steps(args.proposal_learning_rate_decay_end),
        proposal_weight_decay=args.proposal_weight_decay,
        
        radiance_learning_rate_start=args.radiance_learning_rate_start,
        radiance_learning_rate_stop=args.radiance_learning_rate_stop,
        radiance_learning_rate_decay_end=epoch_fraction_to_steps(args.radiance_learning_rate_decay_end),
        radiance_weight_decay=args.radiance_weight_decay,
    )

    # Log model gradients and parameters
    # wandb_logger.watch(model, log="all")

    # Start training
    trainer.fit(model, dm)
