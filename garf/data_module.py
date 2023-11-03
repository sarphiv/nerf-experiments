import os
from copy import copy
from typing import Any, Optional, Literal, cast

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LambdaCallback
import torch as th
from torch.utils.data import DataLoader

from dataset import ImagePoseDataset


class ImagePoseDataModule(pl.LightningDataModule):
    """Data module for loading images from a synthetic dataset.
    Each dataset yields a tuple of (ray_origin, ray_direction, pixel_color).
    All are tensors of shape (3,).
    """
    def __init__(
        self, 
        scene_path: str, 
        image_width: int,
        image_height: int,
        space_transform: Optional[tuple[float, th.Tensor]]=None,
        rotation_noise_sigma: float=1.0,
        translation_noise_sigma: float=1.0,
        noise_seed: Optional[int]=None,
        gaussian_blur_kernel_size: int=40,
        gaussian_blur_relative_sigma_start: float=0.,
        gaussian_blur_relative_sigma_decay: float=1.,
        validation_fraction: float = 1.0,
        validation_fraction_shuffle: Literal["disabled", "random"] | int = "disabled",
        *dataloader_args, **dataloader_kwargs
    ):
        """Initialize the data module.

        Args:
            scene_path (str): Path to the scene directory containing the camera and image data.
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            space_transform (Optional[tuple[float, th.Tensor]], optional): Space transform parameters (cam_max_distance (float), cam_mean (3,)) to apply to the data.
            rotation_noise_sigma (float, optional): Standard deviation of rotation noise in radians. Defaults to 1.0.
            translation_noise_sigma (float, optional): Standard deviation of translation noise. Defaults to 1.0.
            noise_seed (Optional[int], optional): Seed for the noise generator. Defaults to None.
            gaussian_blur_kernel_size (int, optional): Kernel size of the Gaussian blur. Defaults to 40.
            gaussian_blur_relative_sigma_start (float, optional): Starting relative sigma of the Gaussian blur. Defaults to 0..
            gaussian_blur_relative_sigma_decay (float, optional): Relative sigma decay of the Gaussian blur. Defaults to 1..
            validation_fraction (float, optional): Fraction of the validation dataset to use for validation. Defaults to 1.0.
            validation_fraction_shuffle (Literal["disabled", "random"] | int, optional): Whether to shuffle the validation data. 
                If "disabled", validation data is not shuffled. If "random", validation data is shuffled randomly. 
                If an integer, validation data is shuffled using the given random seed. Defaults to "disabled".
            *dataloader_args (Any): Additional arguments to pass to data loaders.
            **dataloader_kwargs (Any): Additional keyword arguments to pass to data loaders.
        """
        super().__init__()
        
        assert 0 <= validation_fraction <= 1, "Validation fraction must be between 0 and 1."

        # Store arguments for dataset
        self.scene_path = scene_path
        self.image_width = image_width
        self.image_height = image_height

        self.space_transform = space_transform

        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.noise_seed = noise_seed
        
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.gaussian_blur_relative_sigma_start = gaussian_blur_relative_sigma_start
        self.gaussian_blur_relative_sigma_decay = gaussian_blur_relative_sigma_decay


        # Store validation dataset splitting arguments
        self.validation_fraction = validation_fraction
        self.validation_fraction_shuffle = validation_fraction_shuffle

        # Store data loader arguments
        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs



    def _get_dataset(self, purpose: Literal["train", "val", "test"]) -> ImagePoseDataset:
        """Get dataset for given purpose.

        Args:
            purpose (Literal["train", "val", "test"]): Purpose of dataset.

        Returns:
            ImagePoseDataset: Dataset for given purpose.
        """
        images_path = os.path.join(self.scene_path, purpose).replace("\\", "/")
        camera_info_path = os.path.join(self.scene_path, f"transforms_{purpose}.json").replace("\\", "/")
        
        dataset = ImagePoseDataset(
            image_width=self.image_width,
            image_height=self.image_height,
            images_path=images_path,
            camera_info_path=camera_info_path,
            space_transform=self.space_transform,
            rotation_noise_sigma=self.rotation_noise_sigma,
            translation_noise_sigma=self.translation_noise_sigma,
            noise_seed=self.noise_seed,
            gaussian_blur_kernel_size=self.gaussian_blur_kernel_size,
            gaussian_blur_relative_sigma_start=self.gaussian_blur_relative_sigma_start,
            gaussian_blur_relative_sigma_decay=self.gaussian_blur_relative_sigma_decay
        )

        return dataset


    def setup(self, stage: Literal["fit", "test", "predict"]):
        self.dataset_train = self._get_dataset("train")
        self.space_transform = self.dataset_train.space_transform


        match stage:
            case "fit":
                self.dataset_val = self._get_dataset("val")

                # Prepare cache of validation data
                self._dataset_val_cache = None
                self._dataset_val_cache_settings = None
                self.val_dataloader()


            case "test":
                self.dataset_test = self._get_dataset("test")


            case "predict":
                pass



    def get_dataset_blur_scheduler_callback(
        self, 
        epoch_fraction_period: float=1.0,
        dataset_name: Literal["train", "val", "test"]="train"
    ) -> LambdaCallback:
        def step(trainer: pl.Trainer, model: pl.LightningModule, batch: th.Tensor, batch_idx: int):
            # Calculate epoch fraction
            epoch_fraction = trainer.current_epoch + batch_idx/trainer.num_training_batches
            
            # If time to step, step gaussian blur in dataset
            if epoch_fraction >= step.schedule_point:
                step.schedule_point += epoch_fraction_period

                match dataset_name:
                    case "train":
                        dataset = trainer.datamodule.dataset_train # type: ignore
                    case "val":
                        dataset = trainer.datamodule.dataset_val # type: ignore
                    case "test":
                        dataset = trainer.datamodule.dataset_test # type: ignore

                dataset = cast(ImagePoseDataset, dataset)
                dataset.gaussian_blur_step()


        # Initialize schedule point
        step.schedule_point = epoch_fraction_period

        # Return callback
        return LambdaCallback(on_train_batch_start=step)



    def _disable_shuffle_arg(self, dataloader_args: tuple, dataloader_kwargs: dict) -> Any:
        dataloader_kwargs = {**dataloader_kwargs}
        
        if len(dataloader_args) > 2:
            dataloader_args = (*dataloader_args[:2], False, *dataloader_args[3:])

        if "shuffle" in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = False

        return dataloader_args, dataloader_kwargs


    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            *self.dataloader_args,
            **self.dataloader_kwargs
        )

    def val_dataloader(self):
        # Disable shuffle on the data loader
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)


        # If dataset has been cached and seeds match, return cached dataset
        if self._dataset_val_cache is not None and self._dataset_val_cache_settings == (self.validation_fraction, self.validation_fraction_shuffle):
            dataset = self._dataset_val_cache
        # Else, create new validation dataset
        else:
            # Get length of validation dataset fraction
            # NOTE: Doing it on a image basis, not a ray basis
            validation_size = int(len(self.dataset_val.dataset) * self.validation_fraction)

            # If validation fraction shuffle is disabled, 
            #  take the first validation fraction of the dataset
            if self.validation_fraction_shuffle == "disabled":
                indices = range(validation_size)
            # Else, shuffle the dataset
            else:
                # Create random number generator
                rng = th.Generator()

                # If seed given, set seed
                if isinstance(self.validation_fraction_shuffle, int):
                    rng.manual_seed(self.validation_fraction_shuffle)

                # Shuffle dataset and split dataset, throwing away the second half
                indices = th.randperm(
                    n=len(self.dataset_val.dataset), 
                    generator=rng
                )[:validation_size].tolist()


            # Get subset of validation dataset
            # NOTE: Shallow copy of dataset
            #  so that the rest of the data is still available
            dataset = copy(self.dataset_val)

            # Retrieve subset of dataset
            dataset.dataset = [dataset.dataset[i] for i in indices]


            # Store processed dataset in cache
            self._dataset_val_cache = dataset
            self._dataset_val_cache_settings = (self.validation_fraction, self.validation_fraction_shuffle)


        # Return data loader of validation dataset
        return DataLoader(
            dataset,
            *args,
            **kwargs
        )

    def test_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)
        return DataLoader(
            self.dataset_test,
            *args,
            **kwargs
        )
