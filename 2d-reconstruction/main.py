import torch as th
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from model import Nerf2d
from data_loader import SingleImageDataModule




dm = SingleImageDataModule(
    # image_path="data/morten-lena.png",
    image_path="data/banana.jpg",
    pixel_shuffle_seed=1337,
    pixel_split_sizes=(0.9, 0.05, 0.05),
    batch_size=256,
    num_workers=1,
    shuffle=True,
)

dm.setup(None)

model = Nerf2d(
    width=dm.image_width, 
    height=dm.image_height, 
    fourier_levels=10,
    learning_rate=1e-4,
    weight_decay=0)

# th.set_float32_matmul_precision("medium")
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=512,
    # precision="16-mixed"
)


plt.ion()
plt.show()


# TODO: Refactor image drawing into a hook
# TODO: The hook should work with wandb logging


trainer.fit(model, dm)

