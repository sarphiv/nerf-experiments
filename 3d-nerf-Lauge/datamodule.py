import torch as tc
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
import typing as tp



class RayDataModule(pl.LightningDataModule):
    def __init__(self, path: str="data/nerf_synthetic/drums", batch_size: int=16):
        """
        DataModule for the rays and their corresponding camera matrices.
        
        Args:
            path (str): Path to the directory containing the data.
            batch_size (int): Batch size - number of rays to train with.
        """

        super().__init__()
        self.transform = transforms.Compose([transforms.PILToTensor()])
        self.path = path
/        self.data_raw
        self.batch_size = batch_size
    
    def get_data_loader(self, stage):
        # return DataLoader(self.images[stage], )
        pass

    def setup(self, stage: str) -> None:
        self.data_raw = {s: self.load_images(os.path.join(self.path, s)) for s in ["train", "test", "val"]}


    def get_rays(self, images: tc.Tensor, cam2world: tc.Tensor, camera_angle_x: float):
        """
        Returns the rays and their corresponding camera matrices.
        
        Args:
            images (tc.Tensor): Images.
            cam2world (tc.Tensor): Camera
        """
        origins = cam2world[:, :3, 3]
        H, W = images.shape[-2:]
        focal = 1 / tc.tan(0.5 * tc.tensor(camera_angle_x))
        directions = tc.meshgrid(tc.linspace(-1,1,H), tc.linspace(-1,1,W))
        directions = tc.flatten(tc.stack(directions), 1)
        directions = tc.cat((directions, -tc.ones_like(directions[:, :1])), dim=-1)



    def load_images(self,
                    file_path: str = None,
                    meta_path: str = None,
                    transform = transforms.PILToTensor()
                    ) -> tp.Tuple[tc.Tensor, tc.Tensor, float]:
        
        """
        Loads images from a file or a directory. If a directory is specified, all images in the directory are loaded.
        If a meta file is specified, the images are loaded from the paths specified in the meta file.

        Args:
            file_path (str, optional): Path to a file or directory. Defaults to None.
            meta_path (str, optional): Path to a meta file. Defaults to None.
            transform (torchvision.transforms, optional): Transform to apply to the images. Defaults to transforms.PILToTensor().

        """
        
        
        assert file_path is not None or meta_path is not None, "Either file_path or meta_path must be specified."
        assert file_path is None or meta_path is None, "Either file_path or meta_path must be specified, not both."

        def _open_image(path) -> tc.Tensor:
            with Image.open(path) as f:
                return transform(f)
            
        # if path is a file
        transform = transforms.PILToTensor()
        if file_path is not None and os.path.isfile(file_path):
            images = _open_image(file_path)
            images = images.unsqueeze(0)

            cam2world = None
            camera_angle_x = None
        
        # if path is a directory
        elif file_path is not None and os.path.isdir(file_path):
            images = []
            for filename in os.listdir(file_path):
                images.append(_open_image(os.path.join(file_path, filename)))

            images = tc.stack(images)

            cam2world = None
            camera_angle_x = None


        # if path is a meta file
        elif meta_path is not None and os.path.isfile(meta_path):

            with open(meta_path) as f:
                meta = json.load(f)

            images = []
            cam2world = []

            for frame in meta["frames"]:
                filename = frame["file_path"][2:] + ".png"
                with Image.open(os.path.join(self.path, filename)) as f:
                    images.append(transform(f))
                cam2world.append(tc.tensor(frame["transform_matrix"]))

            images = tc.stack(images)
            cam2world = tc.stack(cam2world)
            camera_angle_x = meta["camera_angle_x"]

        else:
            raise ValueError("Path is neither an image, a meta data file nor a directory.")
        
        return images, cam2world, camera_angle_x

if __name__ == "__main__":
    # '''Load image and display it.'''
    dm = RayDataModule("data/nerf_synthetic/drums")
    dm.load_images()
    
    image, _, _ = RayDataModule.load_images(None, "data/nerf_synthetic/drums/train/r_0.png")
    print(image.shape)
    plt.imshow(image[0].permute(1, 2, 0))
    plt.show()
