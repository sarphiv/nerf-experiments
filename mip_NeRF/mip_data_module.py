from data_module import ImagePoseDataset, ImagePoseDataModule
import torch as th
import os

from typing import Literal, Tuple

DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor, float]

class MipImagePoseDataset(ImagePoseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixel_width = 1/self.focal_length/self.image_width

    def __getitem__(self, index: int) -> DatasetOutput:
        # Get dataset via image index
        P, o, d, c = self.dataset[index // self.image_batch_size]
        # Get pixel index
        i = index % self.image_batch_size

        return o.view(-1, 3)[i], d.view(-1, 3)[i], c.view(-1, 3, len(self.gaussian_smoothing_sigmas))[i], self.pixel_width



class MipImagePoseDataModule(ImagePoseDataModule):
    def _get_dataset(self, purpose: Literal["train", "val", "test"]) -> ImagePoseDataset:
        """Get dataset for given purpose.

        Args:
            purpose (Literal["train", "val", "test"]): Purpose of dataset.

        Returns:
            ImagePoseDataset: Dataset for given purpose.
        """
        dataset = MipImagePoseDataset(
            image_width=self.image_width,
            image_height=self.image_height,
            images_path=os.path.join(self.scene_path, purpose).replace("\\", "/"),
            pose_path=os.path.join(self.scene_path, f"transforms_{purpose}.json").replace("\\", "/"),
            space_transform=self.space_transform
        )

        return dataset