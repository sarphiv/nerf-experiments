import torch as tc
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import matplotlib.pyplot as plt
import os


def ImageDataModule(pl.LightningDataModule):


def load_images(path: str) -> tc.Tensor:
    """Load image from path and resize to width x height."""
    # check if path is a file
    transform = transforms.PILToTensor()
    if os.path.isfile(path):
        image = Image.open(path)
        image = transform(image)
        image = image.unsqueeze(0)
    # check if path is a directory
    elif os.path.isdir(path):
        image = [transform(Image.open(os.path.join(path, filename))) for filename in os.listdir(path)]
        image = tc.stack(image)
    else:
        raise ValueError("Path is not a file or directory.")
    return image

if __name__ == "__main__":
    '''Load image and display it.'''
    image = load_images("data/nerf_synthetic/drums/train/r_0.png")
    print(image.shape)
    plt.imshow(image[10].permute(1, 2, 0))
    plt.show()
