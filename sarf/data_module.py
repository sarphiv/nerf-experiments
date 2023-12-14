import os
from copy import copy
from typing import Any, Optional, Literal, cast, Union
from itertools import product
import warnings

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
        space_transform_scale: Optional[float]=None,
        space_transform_translate: Optional[th.Tensor]=None,
        rotation_noise_sigma: float=1.0,
        translation_noise_sigma: float=1.0,
        camera_noise_seed: Optional[int]=None,
        validation_fraction: float = 1.0,
        validation_fraction_shuffle: Literal["disabled", "random"] | int = "disabled",
        dataloader_seed: int = 0,
        verbose=False,
        *dataloader_args, **dataloader_kwargs
    ):
        """Initialize the data module.

        Args:
            scene_path (str): Path to the scene directory containing the camera and image data.
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            space_transform_scale (Optional[float], optional): Scale parameter for the space transform. Defaults to None, which auto-calculates based on max distance.
            space_transform_translate (Optional[th.Tensor], optional): Translation parameter for the space transform. Defaults to None, which auto-calculates the mean.
            rotation_noise_sigma (float, optional): Standard deviation of rotation noise in radians. Defaults to 1.0.
            translation_noise_sigma (float, optional): Standard deviation of translation noise. Defaults to 1.0.
            camera_noise_seed (Optional[int], optional): Seed for the camera noise generator. Defaults to None.
            validation_fraction (float, optional): Fraction of the validation dataset to use for validation. Defaults to 1.0.
            validation_fraction_shuffle (Literal["disabled", "random"] | int, optional): Whether to shuffle the validation data. 
                If "disabled", validation data is not shuffled. If "random", validation data is shuffled randomly. 
                If an integer, validation data is shuffled using the given random seed. Defaults to "disabled".
            dataset_seed (int, optional): Seed for the dataset. Defaults to 0.
            *dataloader_args (Any): Additional arguments to pass to data loaders.
            **dataloader_kwargs (Any): Additional keyword arguments to pass to data loaders.
        """
        super().__init__()
        
        assert 0 <= validation_fraction <= 1, "Validation fraction must be between 0 and 1."

        # Store arguments for dataset
        self.scene_path = scene_path
        self.image_width = image_width
        self.image_height = image_height

        self.space_transform_scale = space_transform_scale
        self.space_transform_translate = space_transform_translate

        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.camera_noise_seed = camera_noise_seed
        
        # Store validation dataset splitting arguments
        self.validation_fraction = validation_fraction
        self.validation_fraction_shuffle = validation_fraction_shuffle

        self.verbose = verbose

        # Store data loader arguments
        self.dataloader_seed = dataloader_seed
        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs


    @property
    def n_training_images(self):
        if hasattr(self, "dataset_train"):
            return self.dataset_train.n_images
        elif hasattr(self, "_n_training_images"):
            return self._n_training_images
        else:
            images_path = os.path.join(self.scene_path, "train").replace("\\", "/")
            self._n_training_images = len(os.listdir(images_path))
            return self._n_training_images


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
            space_transform_scale=self.space_transform_scale,
            space_transform_translate=self.space_transform_translate,
            rotation_noise_sigma=self.rotation_noise_sigma,
            translation_noise_sigma=self.translation_noise_sigma,
            noise_seed=None if self.camera_noise_seed is None else self.camera_noise_seed + hash(purpose),
            verbose=self.verbose
        )

        return dataset


    def setup(self, stage: Literal["fit", "test", "predict"]):
        self.dataset_train = self._get_dataset("train")
        self.space_transform_scale = self.dataset_train.space_transform_scale
        self.space_transform_translate = self.dataset_train.space_transform_translate


        match stage:
            case "fit":
                self.dataset_val = self._get_dataset("val")
 
                # Prepare cache of validation dataset fraction
                self._dataset_val_cache = None
                self._dataset_val_cache_settings = None
                self.val_dataloader()


            case "test":
                self.dataset_test = self._get_dataset("test")


            case "predict":
                pass



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
            generator=th.Generator().manual_seed(self.dataloader_seed),
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
            validation_size = int(self.dataset_val.n_images * self.validation_fraction)

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
                    n=self.dataset_val.n_images, 
                    generator=rng
                )[:validation_size].tolist()
            

            # Get subset of validation dataset
            # NOTE: Shallow copy of dataset
            #  so that the rest of the data is still available
            # Retrieve subset of dataset
            dataset = self.dataset_val.subset_dataset(indices) 

            # Store processed dataset in cache
            self._dataset_val_cache = dataset
            self._dataset_val_cache_settings = (self.validation_fraction, self.validation_fraction_shuffle)


        # Return data loader of validation dataset
        return DataLoader(
            dataset,
            generator=th.Generator().manual_seed(self.dataloader_seed),
            *args,
            **kwargs
        )

    def test_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)
        return DataLoader(
            self.dataset_test,
            generator=th.Generator().manual_seed(self.dataloader_seed),
            *args,
            **kwargs
        )
