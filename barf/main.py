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
from model_interpolation_architecture import BarfPositionalEncoding



# High priority 
# TODO: Script that generates runs where the plots are used for the paper 
# TODO: Bug hunting
# TODO: Write paper 

# Low priority
# TODO: make functions in dataset.py static if possible
# TODO: Fix image logger - see that file
# TODO: Remove all the runs in WANDB (davids are gone)
# TODO: Use decay of learningrate - Check if it works in the bottom of model_camera_calibration - for camera extrinsics go as default from 1e-3 to 1e-5. For normal nerf from 5e-4 to 1e-4 
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
    # Parse arguments
    # NOTE: Default is BARF settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="unknown-run-BARF-mebe")
    # parser.add_argument("--run_name", type=str, default="BARF-test-before-hpc")
    parser.add_argument("--rotation_noise", type=float, default=0.0)
    parser.add_argument("--translation_noise", type=float, default=0.0)
    parser.add_argument('--use_fourier', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Whether to use Fourier features or not')
    parser.add_argument('--use_proposal', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Whether to have a proposal network or not')
    parser.add_argument('--delayed_direction', type=bool, action=argparse.BooleanOptionalAction, default=True, help='When the directional input is feed to the network')
    parser.add_argument('--delayed_density', type=bool, action=argparse.BooleanOptionalAction, default=False, help='When the network outputs the density')
    parser.add_argument('--n_segments', type=int, default=2, help='Number of times the positional data is fed to the network')
    parser.add_argument('--n_hidden', type=int, default=4, help='Number of hidden layers')
    # parser.add_argument('--sigmas_for_blur', type=list, default=[0.0], help='Sigmas for the gaussian blur')
    # parser.add_argument('--sigmas_for_blur', type=list, default=[2**(2), 2**(1), 2**(0), 2**(-1), 2**(-2), 0.0], help='Sigmas for the gaussian blur')
    parser.add_argument('--use_blur', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Whether to use blur or not')
    parser.add_argument('--camera_learning_rate_start', type=float, default=1e-5, help='Learning rate for the camera') #This should be 1e-3 if like barf, but that does not work... 
    parser.add_argument('--camera_learning_rate_end', type=float, default=1e-5, help='Learning rate for the camera')
    parser.add_argument('--initial_fourier_features', type=float, default=0.0, help="Active Fourier features initially")
    parser.add_argument('--start_fourier_features_iterations', type=int, default=20000, help="Start increasing the number of fourier features after this many iterations")
    parser.add_argument('--full_fourier_features_iterations', type=int, default=100000, help="Have all fourier features after this many iterations")
    args = parser.parse_args()

    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger",
        name=args.run_name
    )


    # Set up data module
    BATCH_SIZE = 1024*1
    NUM_WORKERS = 8
    IMAGE_SIZE = 400
    SIGMAS_FOR_BLUR = [0.0] if not args.use_blur else [2**(2), 2**(1), 2**(0), 2**(-1), 2**(-2), 0.0]   
    
    dm = ImagePoseDataModule(
        image_width=IMAGE_SIZE,
        image_height=IMAGE_SIZE,
        scene_path="../data/lego",
        validation_fraction=0.06,
        validation_fraction_shuffle=1234,
        gaussian_blur_sigmas = SIGMAS_FOR_BLUR,
        rotation_noise_sigma = args.rotation_noise,
        translation_noise_sigma = args.translation_noise,
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
            LogCameraExtrinsics(
                wandb_logger=wandb_logger,
                logging_start=0.002,
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
                filename='ckpt_epoch={epoch:02d}-val_loss={val_loss:.2f}',
                every_n_epochs=2,
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
    
    # Initialize the positional encoder
    alpha_increase_start_epoch = convert_iterations_to_epochs(args.start_fourier_features_iterations, BATCH_SIZE, len(dm.dataset_train))
    alpha_increase_end_epoch = convert_iterations_to_epochs(args.full_fourier_features_iterations, BATCH_SIZE, len(dm.dataset_train))

    # When no feature encoding is used the positional encoder is set to the identity
    if args.use_fourier:
        positional_encoder = BarfPositionalEncoding(levels=10,
                                                    alpha_start=0,
                                                    alpha_increase_start_epoch=alpha_increase_start_epoch,
                                                    alpha_increase_end_epoch=alpha_increase_end_epoch,
                                                    include_identity=True)
        directional_encoder = BarfPositionalEncoding(levels=4,
                                                     alpha_start=4,
                                                     alpha_increase_start_epoch=alpha_increase_start_epoch,
                                                     alpha_increase_end_epoch=alpha_increase_end_epoch,
                                                     include_identity=True)
    else: 
        positional_encoder = BarfPositionalEncoding(levels=0,
                                                    alpha_start=0,
                                                    alpha_increase_start_epoch=alpha_increase_start_epoch,
                                                    alpha_increase_end_epoch=alpha_increase_end_epoch,
                                                    include_identity=True)
        directional_encoder = BarfPositionalEncoding(levels=0,
                                                     alpha_start=4,
                                                     alpha_increase_start_epoch=alpha_increase_start_epoch,
                                                     alpha_increase_end_epoch=alpha_increase_end_epoch,
                                                     include_identity=True)

    # Set up model
    model = CameraCalibrationModel(
        n_training_images=len(dm.dataset_train.images),
        # camera_learning_rate=5e-4,
        camera_learning_rate=args.camera_learning_rate_start,
        camera_learning_rate_stop_epoch=8,
        camera_learning_rate_decay=0.999,
        camera_learning_rate_period=0.02,
        camera_weight_decay=0.0,
        near_sphere_normalized=1/10,
        far_sphere_normalized=1/3,
        samples_per_ray=64 + 192,
        n_hidden=args.n_hidden,
        hidden_dim=256,
        position_encoder = positional_encoder,
        direction_encoder = directional_encoder,
        max_gaussian_sigma=max(SIGMAS_FOR_BLUR),
        proposal=(args.use_proposal, 64),
        delayed_direction=args.delayed_direction,
        delayed_density=args.delayed_density,
        n_segments=args.n_segments,
        learning_rate=5e-4,
        learning_rate_stop_epoch = 100,
        learning_rate_decay=2**(log2(5e-5/5e-4) / trainer.max_epochs), # type: ignore
        learning_rate_period = 1.0,
        weight_decay=0
    )


    wandb_logger.watch(model, log="all")

    # Start training, resume from checkpoint
    trainer.fit(model, dm)