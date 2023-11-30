import os
from copy import copy
from typing import Any, Optional, Literal, cast, Union
from itertools import product
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LambdaCallback
import torch as th
from torch.utils.data import DataLoader

from dataset import ImagePoseDataset


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
        space_transform_scale: Optional[float]=None,
        space_transform_translate: Optional[th.Tensor]=None,
        rotation_noise_sigma: float=1.0,
        translation_noise_sigma: float=1.0,
        camera_noise_seed: Optional[int]=None,
        gaussian_blur_sigmas: Optional[list[float]]=[0.0], # TODO: change such that default argument is none, and then gaussian blur is disabled
                                                           # ANSWER: has already been done: if sigma is less than 0.25 no blur is applied
        validation_fraction: float = 1.0,
        validation_fraction_shuffle: Literal["disabled", "random"] | int = "disabled",
        verbose=False,
        *dataloader_args, **dataloader_kwargs
    ):
        """Initialize the data module.

        Args:
            scene_path (str): Path to the scene directory containing the camera and image data.
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            space_transform_scale (Optional[float], optional): Scale parameter for the space transform. Defaults to None, which auto-calculates based on max distance.
            space_transform_translate (Optional[th.Tensor], optional): Translation parameter for the space transform. Defaults to None, which auto-calculates the mean.
            rotation_noise_sigma (float, optional): Standard deviation of rotation noise in radians. Defaults to 1.0.
            translation_noise_sigma (float, optional): Standard deviation of translation noise. Defaults to 1.0.
            camera_noise_seed (Optional[int], optional): Seed for the camera noise generator. Defaults to None.
            gaussian_blur_sigmas: (Optional[list[float]], optional): the predetermined standard deviations for the guassian blur.
            validation_fraction (float, optional): Fraction of the validation dataset to use for validation. Defaults to 1.0.
            validation_fraction_shuffle (Literal["disabled", "random"] | int, optional): Whether to shuffle the validation data. 
                If "disabled", validation data is not shuffled. If "random", validation data is shuffled randomly. 
                If an integer, validation data is shuffled using the given random seed. Defaults to "disabled".
            *dataloader_args (Any): Additional arguments to pass to data loaders.
            **dataloader_kwargs (Any): Additional keyword arguments to pass to data loaders.
        """
        super().__init__()
        
        assert 0 <= validation_fraction <= 1, "Validation fraction must be between 0 and 1."

        # Store arguments for dataset
        self.scene_path = scene_path
        self.image_width = image_width
        self.image_height = image_height

        self.space_transform_scale = space_transform_scale
        self.space_transform_translate = space_transform_translate

        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.camera_noise_seed = camera_noise_seed
        
        self.gaussian_blur_sigmas = gaussian_blur_sigmas
        # Store validation dataset splitting arguments
        self.validation_fraction = validation_fraction
        self.validation_fraction_shuffle = validation_fraction_shuffle

        self.verbose = verbose

        # Store data loader arguments
        self.dataloader_args = dataloader_args
        self.dataloader_kwargs = dataloader_kwargs

    @property
    def n_training_images(self):
        if hasattr(self, "dataset_train"):
            return self.dataset_train.n_images
        else:
            images_path = os.path.join(self.scene_path, "train").replace("\\", "/")
            return len(os.listdir(images_path))


    @staticmethod
    def _worker_init_fn(worker_id):
        os.sched_setaffinity(0, range(os.cpu_count())) 

    def _get_dataset(self, purpose: Literal["train", "val", "test"]) -> ImagePoseDataset:
        """Get dataset for given purpose.

        Args:
            purpose (Literal["train", "val", "test"]): Purpose of dataset.

        Returns:
            ImagePoseDataset: Dataset for given purpose.
        """
        images_path = os.path.join(self.scene_path, purpose).replace("\\", "/")
        camera_info_path = os.path.join(self.scene_path, f"transforms_{purpose}.json").replace("\\", "/")
        
        dataset = ImagePoseDataset(
            image_width=self.image_width,
            image_height=self.image_height,
            images_path=images_path,
            camera_info_path=camera_info_path,
            space_transform_scale=self.space_transform_scale,
            space_transform_translate=self.space_transform_translate,
            rotation_noise_sigma=self.rotation_noise_sigma,
            translation_noise_sigma=self.translation_noise_sigma,
            noise_seed=None if self.camera_noise_seed is None else self.camera_noise_seed + hash(purpose),
            gaussian_blur_sigmas = self.gaussian_blur_sigmas,
            verbose=self.verbose
        )

        return dataset


    def setup(self, stage: Literal["fit", "test", "predict"]):
        self.dataset_train = self._get_dataset("train")
        self.space_transform_scale = self.dataset_train.space_transform_scale
        self.space_transform_translate = self.dataset_train.space_transform_translate


        match stage:
            case "fit":
                self.dataset_val = self._get_dataset("val")
                
                # TODO: What is this? 
                # Prepare cache of validation data
                self._dataset_val_cache = None
                self._dataset_val_cache_settings = None
                # self.val_dataloader()


            case "test":
                self.dataset_test = self._get_dataset("test")


            case "predict":
                pass



    def get_dataset_blur_scheduler_callback(
        self, 
        epoch_fraction_period: float=1.0,
        dataset_name: Literal["train", "val", "test"]="train"
    ) -> LambdaCallback:
        def step(trainer: pl.Trainer, model: pl.LightningModule, batch: th.Tensor, batch_idx: int):
            # Calculate epoch fraction
            epoch_fraction = trainer.current_epoch + batch_idx/trainer.num_training_batches
            
            # If time to step, step gaussian blur in dataset
            if epoch_fraction >= step.schedule_point:
                step.schedule_point += epoch_fraction_period

                match dataset_name:
                    case "train":
                        dataset = trainer.datamodule.dataset_train # type: ignore
                    case "val":
                        dataset = trainer.datamodule.dataset_val # type: ignore
                    case "test":
                        dataset = trainer.datamodule.dataset_test # type: ignore

                dataset = cast(ImagePoseDataset, dataset)
                dataset.gaussian_blur_step()


        # Initialize schedule point
        step.schedule_point = epoch_fraction_period

        # Return callback
        return LambdaCallback(on_train_batch_start=step)



    def _disable_shuffle_arg(self, dataloader_args: tuple, dataloader_kwargs: dict) -> Any:
        dataloader_kwargs = {**dataloader_kwargs}
        
        if len(dataloader_args) > 2:
            dataloader_args = (*dataloader_args[:2], False, *dataloader_args[3:])

        if "shuffle" in dataloader_kwargs:
            dataloader_kwargs["shuffle"] = False

        return dataloader_args, dataloader_kwargs


    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            # worker_init_fn=ImagePoseDataModule._worker_init_fn,
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
            # Get length of validation dataset fraction
            # NOTE: Doing it on a image basis, not a ray basis
            validation_size = int(self.dataset_val.n_images * self.validation_fraction)

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
                    n=self.dataset_val.n_images, 
                    generator=rng
                )[:validation_size].tolist()
            

            # Get subset of validation dataset
            # NOTE: Shallow copy of dataset
            #  so that the rest of the data is still available
            # Retrieve subset of dataset
            dataset = self.dataset_val.subset_dataset(indices) 

            # Store processed dataset in cache
            self._dataset_val_cache = dataset
            self._dataset_val_cache_settings = (self.validation_fraction, self.validation_fraction_shuffle)


        # Return data loader of validation dataset
        return DataLoader(
            dataset,
            # worker_init_fn=ImagePoseDataModule._worker_init_fn,
            *args,
            **kwargs
        )

    def test_dataloader(self):
        args, kwargs = self._disable_shuffle_arg(self.dataloader_args, self.dataloader_kwargs)
        return DataLoader(
            self.dataset_test,
            # worker_init_fn=ImagePoseDataModule._worker_init_fn,
            *args,
            **kwargs
        )


    def get_blurred_pixel_colors(self, batch: tuple, sigma: float) -> tuple:
        """
        Compute the interpolation of the blurred pixel colors
        to get the blurred pixel color used for training, while also
        outputting the original pixel color.

        Parameters:
        -----------
            batch: InnerModelBatchInput - the batch of data to transform
            one of the elements is ray_colors of shape (N, n_sigmas, 3)
            sigma: float - the sigma to use for the interpolation
        
        Returns:
        -----------
            batch: InnerModelBatchInput - the transformed batch of data
            now ray_colors is of shape (N, 2, 3)

        
        Details:
        --------

        The batch contains 6 different elements.
        One of the elements contains the ray_colors (this is
        what it is called in _step_helper() etc.).
        ray_colors is a Tensor of shape (N, n_sigmas, 3), where
        N is the number of rays in the batch, and n_sigmas is the number
        of sigmas used for the blurring. 

        This method only modifies the ray_colors element of the batch.
        It is modified to have shape (N, 2, 3), where 
        ray_colors[:, 1, :] are the original pixel colors, and
        ray_colors[:, 0, :] are the blurred pixel colors.

        
        """


        # Unpack batch
        (
            ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            ray_colors_raw,
            img_idx,
            pixel_width
        ) = batch

        # if sigma is small - no blur is applied
        if sigma <= 0.25:
            batch = (ray_origs_raw,
                    ray_origs_pred,
                    ray_dirs_raw,
                    ray_dirs_pred,
                    th.stack([ray_colors_raw[:,-1], ray_colors_raw[:,-1]], dim=1),
                    img_idx,
                    pixel_width)
            
        # if sigma is large - maximal blur is applied
        elif sigma >= max(self.gaussian_blur_sigmas):
            batch = (ray_origs_raw,
                    ray_origs_pred,
                    ray_dirs_raw,
                    ray_dirs_pred,
                    th.stack([ray_colors_raw[:,0], ray_colors_raw[:,-1]], dim=1),
                    img_idx,
                    pixel_width)
            
            # if sigma is too large - warn the user
            if sigma > max(self.gaussian_blur_sigmas): warnings.warn(f"Tried to get blur with sigma {sigma} but used maximal possible: {max(self.gaussian_blur_sigmas)}.")

        # if sigma is in between - interpolate
        else:
            # Find the sigma closest to the given sigma
            index_low = 0
            for index_high, s in enumerate(self.gaussian_blur_sigmas):
                if s < sigma: break
                index_low = index_high  
            
            # ls_1 + (1-l)s_2 = s <=> l = (s - s_2) / (s_1 - s_2)
            interpolation_coefficient = (sigma - self.gaussian_blur_sigmas[index_high]) / (self.gaussian_blur_sigmas[index_low] - self.gaussian_blur_sigmas[index_high] + 1e-8)
            
            # Make interpolation 
            interpolation = ray_colors_raw[:,index_low] * (interpolation_coefficient) + ray_colors_raw[:,index_high] * (1-interpolation_coefficient)
            
            batch = (ray_origs_raw,
                ray_origs_pred,
                ray_dirs_raw,
                ray_dirs_pred,
                th.stack([interpolation, ray_colors_raw[:,-1]], dim=1),
                img_idx,
                pixel_width)
        
        return batch

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    dm = ImagePoseDataModule(
        image_width=400,
        image_height=400,
        space_transform_scale=1.,
        space_transform_translate=th.Tensor([0,0,0]),
        scene_path="../data/lego",
        verbose=True,
        validation_fraction=0.02,
        validation_fraction_shuffle=1234,
        gaussian_blur_sigmas = [16, 4, 1, 0],
        rotation_noise_sigma = 0,#.15,
        translation_noise_sigma = 0,#.15,
        batch_size=10000,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    dm.setup("fit")

    sigmas = [0,1,2,3,4,5,10,16]

    output = []

    for s in sigmas:
        print(s)
        output.append([])
        for batch_og in tqdm(dm.val_dataloader(), desc="plotting images"):
            batch = dm.get_blurred_pixel_colors(batch_og, s)
            output[-1].append(batch[4][:,0])
        output[-1] = th.cat(tuple(output[-1]), 0).view(-1, dm.image_height, dm.image_width, 3)
    
    for i, img in enumerate(range(output[0].shape[0])):
        for j, s in enumerate(sigmas):
            plt.imsave(f"skydmig_{i}_{s}.png", output[j][i].detach().numpy())

