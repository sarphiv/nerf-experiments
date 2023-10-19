import json
import os
import pathlib
import math
from copy import copy
from typing import Any, Callable, Iterator, Optional, Literal, cast

import pytorch_lightning as pl
import torch as th
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import math


DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor]

def exponential_quality_function(start_quality: float, end_epoch: int):
    a = start_quality
    b = -math.log(start_quality)/end_epoch
    # a = -1 + start_quality
    # b = math.log(2-start_quality)/end_epoch
    f = lambda epoch: a*math.exp(b*epoch)
    return f

def linear_quality_function(start_quality: float, end_epoch: int):
    f = lambda epoch: start_quality + (1 - start_quality) * epoch / end_epoch
    return f

class QualityScheduler:
    def __init__(self, quality_function: Callable[[int],float] = lambda epoch: 1.):
        """
        Image quality scheduler
        
        Parameters:
        ----------
        quality_fuction: Callable that maps epoch to a quality, where
                         quality is a float grater than zero. It can be greater than
                         1, but it will be clipped, so it is ignored if
                         it is not in the interval [0,1]
        
        """
        self.quality_function = quality_function
        self.epoch = 0
        self.quality = 0.
    
    def step(self):
        """
        Step the scheduler
        This evaluates and clips the quality function at the current epoch, and then 
        it increments the epoch by one.
        """
        self.quality = max(0, min(1., self.quality_function(self.epoch)))
        self.epoch += 1

class ImagePoseDataset(Dataset[DatasetOutput]):

    @staticmethod
    def transform_alpha_to_white(img: th.Tensor): return img[-1] * img[:3] + (1 - img[-1])
    
    @staticmethod
    def permute_channels(img: th.Tensor): return img.permute(1, 2, 0)

    def __init__(self, image_width: int, image_height: int, images_path: str, pose_path: str, space_transform: Optional[tuple[float, th.Tensor]]=None) -> None:
        super().__init__()
        
        # Store image dimensions
        self.image_height, self.image_width = image_width, image_height
        self.image_batch_size = self.image_width * self.image_height
        self.space_transform = space_transform
        
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
                tv.transforms.Lambda(ImagePoseDataset.transform_alpha_to_white),
                # Permute channels to (H, W, C)
                # WARN: This is against the convention of PyTorch.
                #  Doing it to enable easier batching of rays.
                tv.transforms.Lambda(ImagePoseDataset.permute_channels)
            ])
        )


        # Load each image, transform, and store
        self.images = {pathlib.PurePath(path).stem: self._open_image(path) for path in os.listdir(self.images_path) }


        # Load camera data from json
        camera_data = json.loads(open(self.pose_path).read())
        
        self.focal_length = self.image_width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
        self.camera_to_world: dict[str, th.Tensor] = { 
            pathlib.PurePath(path).stem: th.tensor(camera_to_world) / camera_to_world[-1][-1]
            for frame in camera_data["frames"] 
            for path, rotation, camera_to_world in [frame.values()] 
        }


        # If space transform is given, initialize transform parameters from data
        if self.space_transform is None:
            # Get average position of all cameras (get last columns and average over each entry)
            camera_positions = th.vstack(tuple(self.camera_to_world.values()))[:, -1].reshape(-1, 4)
            camera_average_position = camera_positions.mean(dim=0)
            camera_average_position[-1] = 0

            # Get the maximum distance of any two cameras 
            camera_max_distance: float = 3*th.cdist(camera_positions, camera_positions, compute_mode="donot_use_mm_for_euclid_dist").max().item()
            
            # Define space transform 
            self.space_transform = (camera_max_distance, camera_average_position)


        # Get the space transform matrices 
        (camera_max_distance, camera_average_position) = self.space_transform

        # Only move the offset
        camera_average_position = th.hstack((th.zeros((4,3)), camera_average_position.unsqueeze(1)))

        # Scale the camera distances
        camera_max_distance_matrix = th.ones((4, 4))
        camera_max_distance_matrix[:-1, -1] = camera_max_distance

        # Move origin to average position of all cameras and scale world coordinates by the 3*the maximum distance of any two cameras
        # self.camera_to_world = { image_name: (camera_to_world - camera_average_position)/camera_max_distance_matrix
        #                          for image_name, camera_to_world 
        #                          in self.camera_to_world.items() }

        # Create unit directions (H, W, 3) in camera space
        # NOTE: Initially normalized such that z=-1 via the focal length.
        #  Camera is looking in the negative z direction.
        #  y-axis is also flipped.
        y, x = th.meshgrid(
            -th.linspace(-self.image_height/2, self.image_height/2, self.image_height) / self.focal_length,
            th.linspace(-self.image_width/2, self.image_width/2, self.image_width) / self.focal_length,
            indexing="ij"
        )
        directions = th.stack((x, y, -th.ones_like(x)), dim=-1)
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

        # Get origins directly from camera to world projection
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


    def _open_image(self, path: str) -> th.Tensor:
        """
        Open image at path and transform to tensor.
        And close file afterwards - see https://pillow.readthedocs.io/en/stable/reference/open_files.html
        """
        with Image.open(os.path.join(self.images_path, path)) as img:
            return self.transform(img)


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
        quality_scheduler: QualityScheduler = QualityScheduler(),
        *dataloader_args, **dataloader_kwargs
    ):
        """Initialize the data module.

        Args:
            scene_path (str): Path to the scene directory containing the camera and image data.
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            validation_fraction (float, optional): Fraction of the validation dataset to use for validation. Defaults to 1.0.
            validation_fraction_shuffle (Literal["disabled", "random"] | int, optional): Whether to shuffle the validation data. 
                If "disabled", validation data is not shuffled. If "random", validation data is shuffled randomly. 
                If an integer, validation data is shuffled using the given random seed. Defaults to "disabled".
            quality_scheduler: QualitySchedyler. Scheduler that updates the warmup quality.
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

        self.quality_scheduler = quality_scheduler

        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs
        
        self.space_transform: Optional[tuple[float, th.Tensor]] = None


    def _get_dataset(self, purpose: Literal["train", "val", "test"]) -> ImagePoseDataset:
        """Get dataset for given purpose.

        Args:
            purpose (Literal["train", "val", "test"]): Purpose of dataset.

        Returns:
            ImagePoseDataset: Dataset for given purpose.
        """

        # step the quality_scheduler if a new traning training loop is started 
        # (i.e. _get_dataset is called with purpose "train")
        if purpose == "train":
            self.quality_scheduler.step()
            self._dataset_val_cache = None
            self._dataset_val_cache_settings = None

        dataset = ImagePoseDataset(
            image_width=int(self.quality_scheduler.quality*self.image_width), # use quality from shceduler
            image_height=int(self.quality_scheduler.quality*self.image_height), # use quality from shceduler
            images_path=os.path.join(self.scene_path, purpose).replace("\\", "/"),
            pose_path=os.path.join(self.scene_path, f"transforms_{purpose}.json").replace("\\", "/"),
            space_transform=self.space_transform
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



    def _disable_shuffle_arg(self, dataloader_args: tuple, dataloader_kwargs: dict) -> Any:
        
        dataloader_kwargs = {**dataloader_kwargs}
        
        if len(dataloader_args) > 2:
            dataloader_args = (*dataloader_args[:2], False, *dataloader_args[3:])

        if "shuffle" in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = False

        return dataloader_args, dataloader_kwargs


    def train_dataloader(self):
        self.dataset_train = self._get_dataset('train') #get new traning set
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
            self.dataset_val = self._get_dataset('val') # new validation set, if not val cache
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
        self.dataset_test = self._get_dataset('test') # you know it
        return DataLoader(
            self.dataset_test,
            *args,
            **kwargs
        )
