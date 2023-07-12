import torch as th
import torch.nn as nn
import torchvision as tv
import numpy as np
import pytorch_lightning as pl
from PIL import Image
from typing import Callable, Optional


class SingleImageDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        image_path: str, 
        pixel_shuffle_seed: int, 
        pixel_split_sizes: tuple[float, float, float], 
        *args, **kwargs
    ):
        super().__init__()
        
        assert sum(pixel_split_sizes) == 1.0, "Pixel split sizes must sum to 1.0"

        self.image_path = image_path
        self.pixel_shuffle_seed = pixel_shuffle_seed
        self.pixel_split_sizes = pixel_split_sizes
        self.transform: Callable[[Image.Image], th.Tensor] = tv.transforms.Compose([
            tv.transforms.ToTensor()
        ])
        
        self.args = args
        self.kwargs = kwargs


    def setup(self, stage: Optional[str] = None):
        self.image: th.Tensor = self.transform(Image.open(self.image_path))
        
        self.image_width, self.image_height = self.image.shape[2], self.image.shape[1]

        resolution = min(self.image_width, self.image_height)

        self.image = self.image[:, :resolution, :resolution]
        self.image_width, self.image_height = resolution, resolution


        x, y = th.meshgrid(
            th.arange(resolution), 
            th.arange(resolution),
            indexing="ij"
        )
        x, y = x.flatten(), y.flatten()

        location = th.hstack((
            x.float().unsqueeze(1) / resolution,
            y.float().unsqueeze(1) / resolution
        ))

        location += 1/2/resolution

        assert 0 < location.max() < 1.0


        idx = th.randperm(
            location.shape[0], 
            generator=th.Generator().manual_seed(self.pixel_shuffle_seed)
        )


        split_amounts = [int(x.shape[0]*split) for split in self.pixel_split_sizes]
        idx_train, idx_val, idx_test = th.split(
            idx, 
            [location.shape[0] - sum(split_amounts[1:])] + split_amounts[1:]
        )


        input_train = location[idx_train]
        input_val   = location[idx_val]
        input_test  = location[idx_test]

        output_train = self.image[:, y[idx_train], x[idx_train]].permute(1, 0)
        output_val   = self.image[:, y[idx_val], x[idx_val]].permute(1, 0)
        output_test  = self.image[:, y[idx_test], x[idx_test]].permute(1, 0)

        self.dataset_train = th.utils.data.TensorDataset(input_train, output_train)
        self.dataset_val   = th.utils.data.TensorDataset(input_val,   output_val)
        self.dataset_test  = th.utils.data.TensorDataset(input_test,  output_test)


    def _disable_shuffle_arg(self, args: tuple, kwargs: dict):
        kwargs = {**kwargs}
        
        if len(args) > 2:
            args = (*args[:2], False, *args[3:])

        if "shuffle" in kwargs:
            kwargs["shuffle"] = False

        return args, kwargs

    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.dataset_train,
            *self.args,
            **self.kwargs)

    def val_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.args, self.kwargs)
        return th.utils.data.DataLoader(
            self.dataset_val,
            *args,
            **kwargs)

    def test_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.args, self.kwargs)
        return th.utils.data.DataLoader(
            self.dataset_test,
            *args,
            **kwargs)


if __name__ == "__main__":
    dm = SingleImageDataModule(
        image_path="../data/nature.jpg",
        pixel_shuffle_seed=1337,
        pixel_split_sizes=(0.9, 0.05, 0.05),
        batch_size=256,
        num_workers=4,
        shuffle=True
    )
    dm.setup(None)

    import matplotlib.pyplot as plt

    for x, y in dm.train_dataloader():
        print(x,y)