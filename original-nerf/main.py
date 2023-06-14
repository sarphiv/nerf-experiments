import pytorch_lightning as pl
import torch as th
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger  # type: ignore

from data_module import ImageSyntheticDataModule
from image_logger import Log2dImageReconstruction
from model import NerfOriginal


if __name__ == "__main__":
    # Set seeds
    pl.seed_everything(1337)


    # Set up weights and biases logger
    wandb_logger = WandbLogger(
        project="nerf-experiments", 
        entity="metrics_logger"
    )


    # Set up data module
    
    
    
    
    
    
    
    # TODO: Set up resize transformation
    
    
    
    
    
    
    
    dm = ImageSyntheticDataModule(
        scene_path="data/lego",
        batch_size=1024*4,
        num_workers=4,
        # shuffle=True,
        shuffle=False,
    )
    dm.setup("fit")


    # Set up trainer
    th.set_float32_matmul_precision("medium")
    # th.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator="auto",
        # accelerator="cpu",
        max_epochs=512,
        precision="16-mixed",
        logger=wandb_logger,
        callbacks=[
            Log2dImageReconstruction(
                wandb_logger=wandb_logger,
                epoch_period=1,
                validation_image_name="r_2",
                batch_size=1024*4,
                num_workers=4
            ),
            LearningRateMonitor(
                logging_interval="epoch"
            )
        ]
    )


    # Set up model
    model = NerfOriginal(
        width=dm.image_width, 
        height=dm.image_height, 
        focal_length=dm.focal_length,
        near_sphere_normalized=2,
        far_sphere_normalized=7,
        samples_per_ray=128,
        fourier_levels_pos=10,
        fourier_levels_dir=4,
        learning_rate=5e-4,
        learning_rate_decay=0.5,
        learning_rate_decay_patience=10,
        weight_decay=0
    )


    # Start training
    trainer.fit(model, dm)
