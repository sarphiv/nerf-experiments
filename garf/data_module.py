import os
from copy import copy
from typing import Any, Optional, Literal, cast, Union
from itertools import product

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
        gaussian_blur_sigmas: Optional[list[float]]=[0.0], # TODO: change such that default argument is none, and then gaussian blur is disabled
        validation_fraction: float = 1.0,
        validation_fraction_shuffle: Literal["disabled", "random"] | int = "disabled",
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

        self.space_transform_scale = space_transform_scale
        self.space_transform_translate = space_transform_translate

        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.camera_noise_seed = camera_noise_seed
        
        self.gaussian_blur_sigmas = gaussian_blur_sigmas
        # Store validation dataset splitting arguments
        self.validation_fraction = validation_fraction
        self.validation_fraction_shuffle = validation_fraction_shuffle

        # Store data loader arguments
        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs


    @staticmethod
    def _worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count())) 

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
            gaussian_blur_sigmas = self.gaussian_blur_sigmas,
        )

        return dataset


    def setup(self, stage: Literal["fit", "test", "predict"]):
        self.dataset_train = self._get_dataset("train")
        self.space_transform_scale = self.dataset_train.space_transform_scale
        self.space_transform_translate = self.dataset_train.space_transform_translate


        match stage:
            case "fit":
                self.dataset_val = self._get_dataset("val")
                
                # TODO: What is this? 
                # Prepare cache of validation data
                self._dataset_val_cache = None
                self._dataset_val_cache_settings = None
                # self.val_dataloader()


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
            worker_init_fn=ImagePoseDataModule._worker_init_fn,
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
            worker_init_fn=ImagePoseDataModule._worker_init_fn,
            *args,
            **kwargs
        )

    def test_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)
        return DataLoader(
            self.dataset_test,
            worker_init_fn=ImagePoseDataModule._worker_init_fn,
            *args,
            **kwargs
        )



    # def _get_camera_center_rays(self, dataset: ImagePoseDataset, device: Optional[Union[th.device, str]] = None) -> tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    #     """Get camera center rays from dataset. Rays are ordered by camera index.
        
    #     Args:
    #         dataset (ImagePoseDataset): Dataset to get camera center rays from.
    #         device (Optional[Union[th.device, str]], optional): Device to put camera center rays on. Defaults to None, which uses default device.

    #     Returns:
    #         tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]: Raw and noisy camera center rays (origins, directions).
    #     """
    #     # Get indices to access corners of image
    #     corner_idx = list(zip(*product((0, dataset.image_height-1), range(dataset.image_width))))

    #     def get_center_ray(
    #         origins_store: dict[str, th.Tensor], 
    #         directions_store: dict[str, th.Tensor]
    #     ) -> tuple[th.Tensor, th.Tensor]:
    #         # Retrieve camera focal point
    #         origins = th.vstack([origin[0, 0] for origin in origins_store.values()])
    #         origins.to()

    #         # Retrieve image corners and take their mean to get optical center ray (normalized)
    #         directions = th.stack([direction[corner_idx] for direction in directions_store.values()])
    #         directions = directions.mean(dim=1)
    #         directions = directions / th.norm(directions, dim=1, keepdim=True)

    #         # If device given, move to device
    #         if device is not None:
    #             origins = origins.to(device)
    #             directions = directions.to(device)

    #         # Return center rays for each image
    #         return origins, directions


    #     # Return raw and noisy camera center rays
    #     return (
    #         get_center_ray(dataset.origins_raw, dataset.directions_raw), 
    #         get_center_ray(dataset.origins_noisy, dataset.directions_noisy)
    #     )


    # def train_camera_center_rays(self, device: Optional[Union[th.device, str]] = None) -> tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    #     """Get camera center rays from training dataset. Rays are ordered by camera index.

    #     Args:
    #         device (Optional[Union[th.device, str]], optional): Device to put camera center rays on. Defaults to None, which uses default device.

    #     Returns:
    #         tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]: Raw and noisy camera center rays (origins, directions).
    #     """
    #     return self._get_camera_center_rays(self.dataset_train, device)

    # def val_camera_center_rays(self, device: Optional[Union[th.device, str]] = None) -> tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    #     """Get camera center rays from training dataset. Rays are ordered by camera index.

    #     Args:
    #         device (Optional[Union[th.device, str]], optional): Device to put camera center rays on. Defaults to None, which uses default device.

    #     Returns:
    #         tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]: Raw and noisy camera center rays (origins, directions).
    #     """
    #     return self._get_camera_center_rays(self.dataset_val, device)

    # def test_camera_center_rays(self, device: Optional[Union[th.device, str]] = None) -> tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]:
    #     """Get camera center rays from training dataset. Rays are ordered by camera index.

    #     Args:
    #         device (Optional[Union[th.device, str]], optional): Device to put camera center rays on. Defaults to None, which uses default device.

    #     Returns:
    #         tuple[tuple[th.Tensor, th.Tensor], tuple[th.Tensor, th.Tensor]]: Raw and noisy camera center rays (origins, directions).
    #     """
    #     return self._get_camera_center_rays(self.dataset_test, device)