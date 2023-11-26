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
from model_camera_calibration import CameraCalibrationModel
from model_interpolation_architecture import NerfModel
from positional_encodings import BarfPositionalEncoding, IntegratedFourierFeatures
from model_interpolation import NerfInterpolationOurs, NerfInterpolationNerfacc, uniform_sampling_strategies, integration_strategies
from model_barf import BarfModel
from model_mip import MipNeRF

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform_sampling_strategy", type=str, default="stratified_uniform")
    parser.add_argument("--integration_strategy", type=str, default="middle")
    parser.add_argument("--uniform_sampling_offset_size", type=float, default=-1.)
    parser.add_argument("--use_proposal", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    # print(args)
    print(f"testing integration/sample_strats - proposal={args.use_proposal}, {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}")

    quit()
    exit()

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name="testing mip nerf",
        # name=f"testing integration/sample_strats - proposal={args.use_proposal}, {args.uniform_sampling_strategy}, {args.integration_strategy}, {args.uniform_sampling_offset_size}",
    )


    # Set up data module
    BATCH_SIZE = 1024*2
    NUM_WORKERS = 3
    # IMAGE_SIZE = 200
    IMAGE_SIZE = 400
    SIGMAS_FOR_BLUR = [0.0]
    
    dm = ImagePoseDataModule(
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
        space_transform_scale=1.,
        space_transform_translate=th.Tensor([0,0,0]),
        scene_path="../data/lego",
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        gaussian_blur_sigmas = SIGMAS_FOR_BLUR,
        rotation_noise_sigma = 0,#.15,
        translation_noise_sigma = 0,#.15,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        pin_memory=True
    )

    dm.setup("fit")


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
                delay_start=1/16,
                delay_end=1.,
                delay_taper=5.0,
                train_image_names=["r_1", "r_23"],
                validation_image_names=["r_2", "r_84"],
                reconstruction_batch_size=BATCH_SIZE,
                reconstruction_num_workers=NUM_WORKERS,
                metric_name_val="val_img",
                metric_name_train="train_img",
            ),
            # LogCameraExtrinsics(
            #     wandb_logger=wandb_logger,
            #     logging_start=0.000,
            #     delay_start=1/200,
            #     delay_end=1/16,
            #     delay_taper=4.0,
            #     batch_size=BATCH_SIZE,
            #     num_workers=NUM_WORKERS,
            #     ray_direction_length=1/10,
            #     metric_name="train_point",
            # ),
            LearningRateMonitor(
                logging_interval="step"
            ),
            ModelCheckpoint(
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.3e}',
                every_n_epochs=1,
                save_top_k=-1,
            ),
            # dm.get_dataset_blur_scheduler_callback(
            #     epoch_fraction_period=0.02,
            #     dataset_name="train"
            # ),
            # dm.get_dataset_blur_scheduler_callback(
            #     epoch_fraction_period=0.02,
            #     dataset_name="val"
            # )
        ]
    )


    position_encoder = IntegratedFourierFeatures(levels=10, 
                                                 scale=1.,
                                                 include_identity=True)
    # position_encoder = BarfPositionalEncoding(levels=10,
    #                                             # alpha_start=0,
    #                                             alpha_start=10,
    #                                             alpha_increase_start_epoch=1.28,
    #                                             alpha_increase_end_epoch=6.4,
    #                                             include_identity=True,
    #                                             scale=1.
    #                                             )
    direction_encoder = BarfPositionalEncoding(levels=4,
                                                    alpha_start=4,
                                                    # alpha_start=0,
                                                    alpha_increase_start_epoch=1.28,
                                                    alpha_increase_end_epoch=6.4,
                                                    include_identity=True,
                                                    scale=1.
                                                    )

    model_radiance_and_proposal = NerfModel(
        n_hidden=4,
        hidden_dim=256,
        delayed_direction=True,
        delayed_density=False,
        n_segments=2,
        position_encoder=position_encoder,
        direction_encoder=direction_encoder,
        learning_rate_start=5e-4,
        learning_rate_stop=1e-4,
        learning_rate_decay_end=200000,
    )
    # if args.use_proposal:
    #     model_proposal = NerfModel(
    #         n_hidden=4,
    #         hidden_dim=256,
    #         delayed_direction=True,
    #         delayed_density=False,
    #         n_segments=2,
    #         position_encoder=position_encoder,
    #         direction_encoder=direction_encoder,
    #         learning_rate_start=5e-4,
    #         learning_rate_stop=1e-4,
    #         learning_rate_decay_end=200000,
    #     )
    #     samples_per_ray_proposal=32,

    # else:
    #     model_proposal=None
    #     samples_per_ray_proposal=0,

    model = MipNeRF(
        near_sphere_normalized=2.,
        far_sphere_normalized=8.,
        samples_per_ray_radiance=256,
        samples_per_ray_proposal=64,
        uniform_sampling_strategy="stratified_uniform",
        uniform_sampling_offset_size=-1.,
        integration_strategy="middle",
        # samples_per_ray_radiance=256,
        # samples_per_ray_proposal=64,
        # uniform_sampling_strategy=args.uniform_sampling_strategy,
        # uniform_sampling_offset_size=args.uniform_sampling_offset_size,
        # integration_strategy=args.integration_strategy,
        model_radiance=model_radiance_and_proposal,
        model_proposal=model_radiance_and_proposal,
    )

    # # # Set up model
    # model = BarfModel(
    #     n_training_images=len(dm.dataset_train.images),
    #     camera_learning_rate_start=1e-3,
    #     camera_learning_rate_stop=1e-5,
    #     camera_learning_rate_decay_end=200000,
    #     max_gaussian_sigma=max(SIGMAS_FOR_BLUR),
    #     near_sphere_normalized=2.,
    #     far_sphere_normalized=8.,
    #     samples_per_ray_radiance=128,
    #     # samples_per_ray_proposal=64,
    #     model_radiance=model_radiance,
    #     # model_proposal=model_proposal,
    # )


    # wandb_logger.watch(model, log="all")

    # Start training, resume from checkpoint
    trainer.fit(model, dm)#, ckpt_path="/work3/s204111/nerf-experiments/barf/nerf-experiments/vq9nm9vt/checkpoints/ckpt_epoch=epoch=03-val_loss=val_loss=0.00.ckpt")