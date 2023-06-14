import json
import os
import pathlib
import math
from typing import Any, Callable, Iterator, Literal, cast

import pytorch_lightning as pl
import torch as th
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor]


class ImageSyntheticDataset(Dataset[DatasetOutput]):
    def __init__(self, camera_path: str, images_path: str) -> None:
        super().__init__()
        
        self.camera_path = camera_path
        self.images_path = images_path
        
        self.transform = cast(
            Callable[[Image.Image], th.Tensor], 
            tv.transforms.Compose([
                # Transform form PIL image to Tensor
                tv.transforms.ToTensor(),
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
        
        # Store image dimensions
        self.height, self.width = next(iter(self.images.values())).shape[:2]
        self.image_batch_size = self.width * self.height


        # Load camera data from json
        camera_data = json.loads(open(self.camera_path).read())
        
        self.focal_length = self.width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
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
            -th.linspace(-self.height/2, self.height/2, self.height) / self.focal_length,
            th.linspace(-self.width/2, self.width/2, self.width) / self.focal_length,
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



class ImageSyntheticDataModule(pl.LightningDataModule):
    """Data module for loading images from a synthetic dataset.
    Each dataset yields a tuple of (ray_origin, ray_direction, pixel_color).
    All are tensors of shape (3,).
    """
    def __init__(
        self, 
        scene_path: str, 
        *args, **kwargs
    ):
        super().__init__()

        self.scene_path = scene_path
        
        self.image_width: int = 0
        self.image_height: int = 0
        self.focal_length: float = 0.0
        
        self.args = args
        self.kwargs = kwargs


    def setup(self, stage: Literal["fit", "test", "predict"]):
        match stage:
            case "fit":
                self.dataset_train = ImageSyntheticDataset(
                    camera_path=os.path.join(self.scene_path, "transforms_train.json"),
                    images_path=os.path.join(self.scene_path, "train-tiny")
                )
                self.dataset_val = ImageSyntheticDataset(
                    camera_path=os.path.join(self.scene_path, "transforms_val.json"),
                    images_path=os.path.join(self.scene_path, "val-tiny")
                )
                
                self.image_width, self.image_height = self.dataset_train.width, self.dataset_train.height
                self.focal_length = self.dataset_train.focal_length


            case "test":
                self.dataset_test = ImageSyntheticDataset(
                    camera_path=os.path.join(self.scene_path, "transforms_test.json"),
                    images_path=os.path.join(self.scene_path, "test-tiny")
                )
                
                self.image_width, self.image_height = self.dataset_test.width, self.dataset_test.height
                self.focal_length = self.dataset_test.focal_length


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
        args, kwargs = self._disable_shuffle_arg(self.args, self.kwargs)
        return DataLoader(
            self.dataset_val,
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