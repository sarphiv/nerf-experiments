import json
import os
import pathlib
import math
from copy import copy
from typing import Any, Callable, Iterator, Literal, cast

import pytorch_lightning as pl
import torch as th
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor]


class ImagePoseDataset(Dataset[DatasetOutput]):
    def __init__(self, image_width: int, image_height: int, images_path: str, pose_path: str) -> None:
        super().__init__()
        
        # Store image dimensions
        self.image_height, self.image_width = image_width, image_height
        self.image_batch_size = self.image_width * self.image_height
        
        # Store paths
        self.images_path = images_path
        self.pose_path = pose_path

        # Transform from PIL image to Tensor
        self.transform = cast(
            Callable[[Image.Image], th.Tensor], 
            tv.transforms.Compose([
                # Transform form PIL image to Tensor
                tv.transforms.ToTensor(),
                # Resize image
                tv.transforms.Resize((self.image_height, self.image_width), antialias=True), # type: ignore
                # Transform alpha to white background (removes alpha too)
                tv.transforms.Lambda(lambda img: img[-1] * img[:3] + (1 - img[-1])),
                # Permute channels to (H, W, C)
                # WARN: This is against the convention of PyTorch.
                #  Doing it to enable easier batching of rays.
                tv.transforms.Lambda(lambda img: img.permute(1, 2, 0))
            ])
        )


        # Load each image, transform, and store
        self.images = { pathlib.PurePath(path).stem: self.transform(
                            Image.open(str(pathlib.PurePath(images_path, path)))
                        )
                        for path in os.listdir(self.images_path) }


        # Load camera data from json
        camera_data = json.loads(open(self.pose_path).read())
        
        self.focal_length = self.image_width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
        self.camera_to_world: dict[str, th.Tensor] = { 
            pathlib.PurePath(path).stem: th.tensor(camera_to_world) / camera_to_world[-1][-1]
            for frame in camera_data["frames"] 
            for path, rotation, camera_to_world in [frame.values()] 
        }


        # Create unit directions (H, W, 3) in camera space
        # NOTE: Initially normalized such that z=-1 via the focal length.
        #  Camera is looking in the negative z direction.
        #  y-axis is also flipped.
        i, j = th.meshgrid(
            -th.linspace(-self.image_height/2, self.image_height/2, self.image_height) / self.focal_length,
            th.linspace(-self.image_width/2, self.image_width/2, self.image_width) / self.focal_length,
            indexing="ij"
        )
        directions = th.stack((j, i, -th.ones_like(j)), dim=-1)
        directions /= th.norm(directions, p=2, dim=-1, keepdim=True)

        # Rotate directions (H, W, 3) to world via R (3, 3).
        #  Denote dir (row vector) as one of the directions in the directions tensor.
        #  Then R @ dir.T = (dir @ R.T).T. 
        #  This would yield a column vector as output. 
        #  To get a row vector as output again, simply omit the last transpose.
        #  The inside of the parenthesis on the right side 
        #  is conformant for matrix multiplication with the directions tensor.
        # NOTE: Assuming scale of projection matrix is 1
        self.directions: dict[str, th.Tensor] = { 
            image_name: directions @ camera_to_world[:3, :3].T
            for image_name, camera_to_world in self.camera_to_world.items()
        }

        # Get directions directly from camera to world projection
        self.origins = {
            image_name: camera_to_world[:3, 3].expand_as(directions)
            for image_name, camera_to_world in self.camera_to_world.items()
        }


        # Store dataset output
        self.dataset = [
            (
                camera_to_world, 
                self.origins[image_name],
                self.directions[image_name], 
                self.images[image_name]
            ) 
            for image_name, camera_to_world in self.camera_to_world.items()
        ]


    def __getitem__(self, index: int) -> DatasetOutput:
        # Get dataset via image index
        P, o, d, c = self.dataset[index // self.image_batch_size]
        # Get pixel index
        i = index % self.image_batch_size

        return o.view(-1, 3)[i], d.view(-1, 3)[i], c.view(-1, 3)[i]

    def __len__(self) -> int:
        return len(self.dataset) * self.image_batch_size



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
        validation_fraction: float = 1.0,
        validation_fraction_shuffle: Literal["disabled", "random"] | int = "disabled",
        *args, **kwargs
    ):
        """Initialize the data module.

        Args:
            scene_path (str): Path to the scene directory containing the camera and image data.
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            validation_fraction (float, optional): Fraction of the dataset to use for validation. Defaults to 1.0.
            validation_fraction_shuffle (Literal["disabled", "random"] | int, optional): Whether to shuffle the validation data. 
                If "disabled", validation data is not shuffled. If "random", validation data is shuffled randomly. 
                If an integer, validation data is shuffled using the given random seed. Defaults to "disabled".
            *args (Any): Additional arguments to pass to data loaders.
            **kwargs (Any): Additional keyword arguments to pass to data loaders.
        """
        super().__init__()
        
        assert 0 <= validation_fraction <= 1, "Validation fraction must be between 0 and 1."

        self.scene_path = scene_path
        self.image_width = image_width
        self.image_height = image_height

        self.validation_fraction = validation_fraction
        self.validation_fraction_shuffle = validation_fraction_shuffle

        self.args = args
        self.kwargs = kwargs


    def setup(self, stage: Literal["fit", "test", "predict"]):
        match stage:
            case "fit":
                self.dataset_train = ImagePoseDataset(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    images_path=os.path.join(self.scene_path, "train"),
                    pose_path=os.path.join(self.scene_path, "transforms_train.json"),
                )
                self.dataset_val = ImagePoseDataset(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    images_path=os.path.join(self.scene_path, "val"),
                    pose_path=os.path.join(self.scene_path, "transforms_val.json"),
                )

                # Prepare cache of validation data
                self._dataset_val_cache = None
                self._dataset_val_cache_settings = None
                self.val_dataloader()


            case "test":
                self.dataset_test = ImagePoseDataset(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    images_path=os.path.join(self.scene_path, "test"),
                    pose_path=os.path.join(self.scene_path, "transforms_test.json"),
                )


            case "predict":
                pass



    def _disable_shuffle_arg(self, args: tuple, kwargs: dict) -> Any:
        kwargs = {**kwargs}
        
        if len(args) > 2:
            args = (*args[:2], False, *args[3:])

        if "shuffle" in kwargs:
            kwargs["shuffle"] = False

        return args, kwargs


    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            *self.args,
            **self.kwargs
        )

    def val_dataloader(self):
        # Disable shuffle on the data loader
        args, kwargs = self._disable_shuffle_arg(self.args, self.kwargs)


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
            print(indices)
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
        args, kwargs = self._disable_shuffle_arg(self.args, self.kwargs)
        return DataLoader(
            self.dataset_test,
            *args,
            **kwargs
        )