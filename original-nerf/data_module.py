import json
import os
import pathlib
import math
from typing import Any, Callable, Iterator, Literal, cast

import pytorch_lightning as pl
import torch as th
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset


DatasetOutput = tuple[th.Tensor, th.Tensor]


class ImageSyntheticDataLoader(IterableDataset[DatasetOutput]):
    def __init__(self, camera_path: str, images_path: str) -> None:
        super().__init__()
        
        self.camera_path = camera_path
        self.images_path = images_path
        
        self.transform = cast(
            Callable[[Image.Image], th.Tensor], 
            tv.transforms.Compose([
                tv.transforms.ToTensor()
            ])
        )


        # Load each image, transform, and store
        self.images = { pathlib.PurePath(path).stem:  self.transform(
                            Image.open(str(pathlib.PurePath(images_path, path)))
                        )
                        for path in os.listdir(self.images_path) }
        
        # Store image dimensions
        self.width, self.height = next(iter(self.images.values())).shape[-2:]


        # Load camera data from json
        camera_data = json.loads(open(self.camera_path).read())
        
        self.focal_length = self.width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
        self.camera_to_world = { pathlib.PurePath(path).stem: th.tensor(camera_to_world) 
                                 for frame in camera_data["frames"] 
                                 for path, rotation, camera_to_world in [frame.values()] }

        # Store inverse projection matrix and image
        self.dataset = [(self.camera_to_world[image_name], image) 
                        for image_name, image in self.images.items()]



    def __getitem__(self, index: int) -> DatasetOutput:
        return self.dataset[index]

    def __iter__(self) -> Iterator[DatasetOutput]:
        return iter(self.dataset)



class ImageSyntheticDataModule(pl.LightningDataModule):
    """Data module for loading images from a synthetic dataset.
    Each dataset yields a tuple of (camera_to_world, image),
    where camera_to_world is a tensor of shape (4, 4),
    and image is a tensor of shape (4, height, width).
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
                self.dataset_train = ImageSyntheticDataLoader(
                    camera_path=os.path.join(self.scene_path, "transforms_train.json"),
                    images_path=os.path.join(self.scene_path, "train-tiny")
                )
                self.dataset_val = ImageSyntheticDataLoader(
                    camera_path=os.path.join(self.scene_path, "transforms_val.json"),
                    images_path=os.path.join(self.scene_path, "val-tiny")
                )
                
                self.image_width, self.image_height = self.dataset_train.width, self.dataset_train.height
                self.focal_length = self.dataset_train.focal_length


            case "test":
                self.dataset_test = ImageSyntheticDataLoader(
                    camera_path=os.path.join(self.scene_path, "transforms_test.json"),
                    images_path=os.path.join(self.scene_path, "test-tiny")
                )
                
                self.image_width, self.image_height = self.dataset_test.width, self.dataset_test.height
                self.focal_length = self.dataset_test.focal_length


            case "predict":
                pass


    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            *self.args,
            **self.kwargs
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            *self.args,
            **self.kwargs
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            *self.args,
            **self.kwargs
        )