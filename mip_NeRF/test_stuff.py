
from tqdm import tqdm
from data_module import ImagePoseDataset, ImagePoseDataModule
from torch.utils.data import DataLoader, TensorDataset
from typing import cast
import torch as th
from mip_model import MipNerf
from math import log2

dm = ImagePoseDataModule(
    image_width=800,
    image_height=800,
    scene_path="../data/lego",
    validation_fraction=0.05,
    validation_fraction_shuffle=1234,
    batch_size=2048,
    num_workers=4,
    shuffle=True,
)

dm.setup("fit")
dataset = dm.dataset_val

model = MipNerf(
    near_sphere_normalized=1/10,
    far_sphere_normalized=1/3,
    samples_per_ray=64 + 192,
    n_hidden=4,
    fourier=(True, 10, 4),
    proposal=(True, 64),
    n_segments=2,
    learning_rate=5e-4,
    learning_rate_decay=2**(log2(5e-5/5e-4) / 10), # type: ignore
    weight_decay=0
)


class dummy:
    def __init__(self):
        self.device = th.device("cpu")
        self.dtype = th.float32
        self.validation_image_names = ["r_2", "r_84"]
        self.batch_size = 2048
        self.num_workers = 4


self = dummy()

images = []

# Reconstruct each image
for name in tqdm(self.validation_image_names, desc="Reconstructing images", leave=False):
    # Set up data loader for validation image
    data_loader = DataLoader(
        dataset=TensorDataset(
            dataset.origins[name].view(-1, 3), 
            dataset.directions[name].view(-1, 3)
        ),
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        shuffle=False
    )

    pixel_width = th.tensor(dataset.pixel_width, dtype=dataset.origins[name].dtype, device=model.device)
    pixel_width = pixel_width.view(1,1).expand(self.batch_size, 1)

    # Iterate over batches of rays to get RGB values
    rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, model.dtype))
    i = 0
    
    for ray_origs, ray_dirs in tqdm(data_loader, desc="Predicting RGB values", leave=False):
        # Prepare for model prediction
        ray_origs = ray_origs.to(model.device)
        ray_dirs = ray_dirs.to(model.device)
        
        # Get size of batch
        batch_size = ray_origs.shape[0]
        
        # Predict RGB values
        rgb[i:i+batch_size, :] = model(ray_origs, ray_dirs, pixel_width[:batch_size])[0].clip(0, 1).cpu()
    
        # Update write head
        i += batch_size


    # Store image on CPU
    # NOTE: Cannot pass tensor as channel dimension is in numpy format
    images.append(rgb.view(dataset.image_height, dataset.image_width, 3).numpy())

