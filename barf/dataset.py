import json
import os
import pathlib
import math
import sys
from typing import Callable, Optional, cast
import copy

import torch as th
from torch.utils.data import Dataset
import torchvision as tv

from model_camera_extrinsics import CameraExtrinsics



# Type alias for dataset output
#  (origin_raw, origin_noisy, direction_raw, direction_noisy, pixel_color_raw, image_index)
DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]


# TODO: gaussian blur: replace all that gaussian with list of
# sigmas - and then we compute it in discrete steps, and 
# then it is computed by interpolating (in the model probably)

# TODO: Fix image logger - see that file
# TODO: Point cloud 
# TODO: Look through data_module -> Why can't we do the same for
#       in the val_dataloader(). the validation set as for the training set? TORBEN!!!!

# TODO make functions static if possible

class ImagePoseDataset(Dataset[DatasetOutput]):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        images_path: str,
        camera_info_path: str,
        space_transform_scale: Optional[float]=None,
        space_transform_translate: Optional[th.Tensor]=None,
        rotation_noise_sigma: float=1.0,
        translation_noise_sigma: float=1.0,
        noise_seed: Optional[int]=None,
        gaussian_blur_kernel_size: int=40,
        gaussian_blur_relative_sigma_start: float=0.,
        gaussian_blur_relative_sigma_decay: float=1.
    ) -> None:
        """Loads images, camera info, and generates rays for each pixel in each image.
        
        Args:
            image_width (int): Width to resize images to.
            image_height (int): Height to resize images to.
            images_path (str): Path to the image directory.
            camera_info_path (str): Path to the camera info file.
            space_transform_scale (Optional[float], optional): Scale parameter for the space transform. Defaults to None, which auto-calculates based on max distance.
            space_transform_translate (Optional[th.Tensor], optional): Translation parameter for the space transform. Defaults to None, which auto-calculates the mean.
            rotation_noise_sigma (float, optional): Sigma parameter for the rotation noise in radians. Defaults to 1.0.
            translation_noise_sigma (float, optional): Sigma parameter for the translation noise. Defaults to 1.0.
            noise_seed (Optional[int], optional): Seed for the noise generator. Defaults to None.
            [deprecated] gaussian_blur_kernel_size (int, optional): Size of the gaussian blur kernel. Defaults to 5.
            [deprecated] gaussian_blur_relative_sigma_start (float, optional): Initial sigma parameter for the gaussian blur. Set to 0 to disable. Defaults to 0..
            [deprecated] gaussian_blur_relative_sigma_decay (float, optional): Decay factor for the gaussian blur sigma. Defaults to 1..
        """
        super().__init__()

        # Store image dimensions
        self.image_height, self.image_width = image_width, image_height
        self.image_batch_size = self.image_width * self.image_height

        # Store paths
        self.images_path = images_path
        self.camera_info_path = camera_info_path

        # Store noise parameters
        # NOTE: Rotation is measured in radians
        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.noise_seed = noise_seed

        # Load images
        self.images = self._load_images(
            self.images_path, 
            self.image_width, 
            self.image_height
        )

        self.n_images = len(self.images)

        # This is used for subsetting the dataset
        # this is a dict that is used to get the original index of an image in the origina dataset
        # from its index in the current subset of the original dataset:
        #   Key: image index in current subset of original dataset, i.e. and integer in [0, n_images_in_current_dataset)
        #   Value: the original index of the image, i.e. an integer in [0, n_train_images_originally)
        # The reason for this is that we need to be able to remember the original index of the image
        # when we subset the dataset, so that we can get the correct camera extrinsics
        # I.e this dict is only strictly necassary for training images.
        # it is made specifically for the image logger for logging training images.
        self.index_to_index = {i: i for i in range(self.n_images)}

        # Load camera info
        self.focal_length, self.camera_to_worlds = self._load_camera_info(
            self.camera_info_path, 
            self.image_width
        )

        # Transform camera to world matrices
        (
            self.camera_to_worlds, 
            self.space_transform_scale, 
            self.space_transform_translate,
            self.image_name_to_index,
            self.image_index_to_name,
        ) = self._transform_camera_to_world(
            self.camera_to_worlds, 
            space_transform_scale,
            space_transform_translate
        )

        # Store the origins of the cameras and the direction their center is facing
        self.camera_origins, self.camera_directions = self._get_cam_origs_and_directions(self.camera_to_worlds)

        # Create mesh grid for the directions 
        meshgrid = self._get_directions_meshgrid(self.image_width, self.image_height, self.focal_length)
        
        # Transform the unit directions to a direction for each camera
        self.ray_origins, self.ray_directions = self._meshgrid_to_world(meshgrid, self.camera_to_worlds)
        
        # Apply noise to get the input data for the model (both rays origins and directions and the centers)
        (
            self.camera_origins_noisy,
            self.camera_directions_noisy,
            self.ray_origins_noisy,
            self.ray_directions_noisy,
            ) = self._apply_noise(self.camera_origins,
                                  self.camera_directions,
                                  self.ray_origins,
                                  self.ray_directions,
                                  self.rotation_noise_sigma,
                                  self.translation_noise_sigma)

        # TODO TODO TODO!!!!
        # THIS IS ONLY FOR TESTING THE VALIDATION_TRANSFORM FUNCTION IN THE CAMERA CALIBRATION MODEL
        # REMOVE THIS WHEN WE HAVE SEEN THE OTHER THING WORK - ASSS FAAASSST AASSS POOOSSSIIIBBBLLLEEE!!!!
        
        # self._screw_up_original_camera_poses_for_testing_validation_transform_in_CameraCalibrationModel()


    # TODO: Change this such that it produces gaussian blurs in discrete steps. 
    def _load_images(self, image_dir_path, image_width, image_height) -> dict[str, th.Tensor]:

        """
        
        
        Returns
        --------
        images: Tensor(N, H, W, n_sigmas, 3)
        
        """
        
        # TODO: remove and relplace with similar PIL functions
        # Transform image to correct format
        transform = cast(
            Callable[[th.Tensor], th.Tensor], 
            tv.transforms.Compose([
                # Convert to float
                tv.transforms.Lambda(lambda img: img.float() / 255.),
                # Resize image
                tv.transforms.Resize(
                    (image_height, image_width), 
                    interpolation=tv.transforms.InterpolationMode.BICUBIC, 
                    antialias=True # type: ignore
                ),
                # Transform alpha to white background (removes alpha too)
                tv.transforms.Lambda(lambda img: img[-1] * img[:3] + (1 - img[-1])),
                # Permute channels to (H, W, C)
                # WARN: This is against the convention of PyTorch.
                #  Doing it to enable easier batching of rays.
                tv.transforms.Lambda(lambda img: img.permute(1, 2, 0))
            ])
        )
        
        # TODO: load images with the old _open_image function to PIL images
        # Open RGBA image
        read = lambda path: tv.io.read_image(
            os.path.join(image_dir_path, path), 
            tv.io.ImageReadMode.RGB_ALPHA
        )

        # Load each image, transform, and store
        return {
            pathlib.PurePath(path).stem: transform(read(path))
            for path in os.listdir(image_dir_path) 
        }


    def _load_camera_info(self, camera_info_path: str, image_width: int) -> tuple[float, dict[str, th.Tensor]]:
        # Read info file
        camera_data = json.loads(open(camera_info_path).read())
        
        # Calculate focal length from camera horizontal angle
        focal_length = image_width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
        # Get camera to world matrices
        # NOTE: Projections are scaled to have scale 1
        camera_to_worlds: dict[str, th.Tensor] = { 
            pathlib.PurePath(path).stem: th.tensor(c2w) / c2w[-1][-1]
            for frame in camera_data["frames"] 
            for path, rotation, c2w in [frame.values()] 
        }

        # Return focal length and camera to world matrices
        return focal_length, camera_to_worlds


    def _transform_camera_to_world(self, camera_to_worlds: dict[str, th.Tensor], space_transform_scale: Optional[float], space_transform_translate: Optional[th.Tensor]) -> tuple[th.Tensor, float, th.Tensor, dict[str, int], dict[str, int]]:
        # If space transform is not given, initialize transform parameters from data
        # NOTE: Assuming camera_to_world has scale 1
        camera_positions = th.stack(tuple(camera_to_worlds.values()))[:, :3, -1] 

        # If no scale is given, initialize to 3*the maximum distance of any two cameras
        if space_transform_scale is None:
            space_transform_scale = 3*th.cdist(camera_positions, camera_positions, compute_mode="donot_use_mm_for_euclid_dist").max().item()

        # If no translation is given, initialize to mean
        if space_transform_translate is None:
            space_transform_translate = camera_positions.mean(dim=0)


        # Only move the offset
        translate_matrix = th.cat((space_transform_translate, th.zeros(1))).view(4, 1)
        translate_matrix = th.hstack((th.zeros((4,3)), translate_matrix))

        # Scale the camera distances
        scale_matrix = th.ones((4, 4))
        scale_matrix[:-1, -1] = space_transform_scale

        # Move origin to average position of all cameras and scale world coordinates
        # by the 3*the maximum distance of any two cameras
        return (
            th.stack([(c2w - translate_matrix)/scale_matrix 
                      for image_name, c2w in camera_to_worlds.items()],
                      dim = 0
                      ),
            space_transform_scale,
            space_transform_translate,
            { 
                image_name: i
                for i, (image_name, c2w)
                in enumerate(camera_to_worlds.items())
            },
            { 
                i: image_name
                for i, (image_name, c2w)
                in enumerate(camera_to_worlds.items())
            }
        )


    def _get_cam_origs_and_directions(self, camera_to_worlds: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Store the origins of the cameras and the direction their center is facing"""

        camera_origins = camera_to_worlds[:, :3, 3] # th.stack([c2w[:3, 3] for c2w in camera_to_worlds.values()], dim=0)
        camera_directions = th.matmul(camera_to_worlds[:, :3, :3], th.tensor([0,0,-1.]).view(1,3,1)).squeeze(-1) # th.stack([th.tensor([0., 0., -1.])@c2w[:3, :3].T for c2w in camera_to_worlds.values()], dim=0)# old code: 

        return camera_origins, camera_directions


    def _get_directions_meshgrid(self, image_width: int, image_height: int, focal_length: float) -> th.Tensor:
        """
        Creates a generic meshgrid of directions that can be transformed to be the direction for any camera by applying the camera to world matrix.
        Returns the meshgrid flattened to (H*W, 3).
        """
        # Create unit directions (H, W, 3) in camera space
        # NOTE: Initially normalized such that z=-1 via the focal length.
        #  Camera is looking in the negative z direction.
        #  y-axis is also flipped.
        y, x = th.meshgrid(
            -th.linspace(-(image_height-1)/2, (image_height-1)/2, image_height) / focal_length,
            th.linspace(-(image_width-1)/2, (image_width-1)/2, image_width) / focal_length,
            indexing="ij"
        )
        directions = th.stack((x, y, -th.ones_like(x)), dim=-1)
        directions /= th.norm(directions, p=2, dim=-1, keepdim=True)

        return directions.view(-1,3)


    def _meshgrid_to_world(self, meshgrid: th.Tensor, camera_to_worlds: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Transform from the unit meshgrid of directions to a direction for each camera by multiplying with the camera to world matrix.

        Parameters:
        ---------
            meshgrid:  shape = (image_height * image_width, 3), output from _get_directions_meshgrid()
        Returns:
        ---------
            ray_origins: (N, H*W, 3)
            ray_directions: (N, H*W, 3)

        NOTE: The final calculation matches the two dimensions that are not shared with an "empty" dimension: 
              camera_to_worlds: (N, 4, 4)   -> unsqueeze(1)                     -> (N, 1, 4, 4)
              meshgrid:         (H*W, 3)    -> unsqueeze(0) and unsqueeze(-1)   -> (1, H*W, 3, 1)
              Hence the output is (N, H*W, 3, 1) -> squeeze(-1)                 -> (N, H*W, 3)
        """
        return (
            camera_to_worlds[:, :3, 3].unsqueeze(1).repeat(1, meshgrid.shape[0],1),
            th.matmul(camera_to_worlds[:, :3, :3].unsqueeze(1), meshgrid.unsqueeze(0).unsqueeze(-1)).squeeze(-1) 
        )

    # TODO TODO TODO!!!!
    # this functoion is only for testing - should probably be removed when we have seen the other thing work
    def _screw_up_original_camera_poses_for_testing_validation_transform_in_CameraCalibrationModel(
            self) -> th.Tensor:
        # apply some R and t to the camera origins and directions - to test the validation_transform function in CameraCalibrationModel
        R = CameraExtrinsics.so3_to_SO3(th.tensor([23,11.,31.])).squeeze(0)
        R_inv = R.T.unsqueeze(0)
        # gives the rotation matrix:
        #       [-0.1838, -0.2228,  0.9574]
        #  R =  [ 0.7764, -0.6302,  0.0024]
        #       [ 0.6028,  0.7438,  0.2888]
        t = th.tensor([[7., 2., -11.]])
        c = th.tensor(3.6)

        # noisy_poses = R@original_poses*c + t
        # original_poses = R.T @ (noisy_poses - t) / c

        # apply the rotation and translation to the camera origins and directions
        self.camera_directions = th.matmul(R_inv, self.camera_directions.unsqueeze(-1)).squeeze(-1)
        self.camera_origins = th.matmul(R_inv, (self.camera_origins - t).unsqueeze(-1)).squeeze(-1) / c

        self.ray_directions = th.matmul(R_inv.unsqueeze(1), self.ray_directions.unsqueeze(-1)).squeeze(-1)
        self.ray_origins = th.matmul(R_inv.unsqueeze(1), (self.ray_origins - t).unsqueeze(-1)).squeeze(-1) / c

    
    def _apply_noise(self,
                     camera_origins: th.Tensor,
                     camera_directions: th.Tensor,
                     ray_origins: th.Tensor,
                     ray_directions: th.Tensor,
                     rotation_noise_sigma: th.Tensor,
                     translation_noise_sigma: th.Tensor,
                     noise_seed: Optional[int]=None,
                     ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """ 
        Apply noise to the camera origins (a translate) and to the directions (a rotation)
        """

        # Set seed for noise generator
        noise_generator = th.Generator()
        if noise_seed is not None:
            noise_generator.manual_seed(noise_seed)

        # Create N random rotation matrices with the given noise level 
        rotation_noise = CameraExtrinsics.so3_to_SO3(
            th.randn((len(camera_origins), 3, 1), generator=noise_generator)*rotation_noise_sigma,
        )

        # NOTE: This is the old code, but I didn't wanna remove it yes
        # rotation_noise = th.matrix_exp(th.cross(
        #     -th.eye(3).view(1, 3, 3), 
        #     th.randn((len(self.images), 3, 1), generator=noise_generator)*rotation_noise_sigma,
        #     dim=1
        # ))
        
        # Create N random translation vectors with the given noise level
        translation_noise = th.randn((len(camera_origins), 3))*translation_noise_sigma

        # Apply translations 
        camera_origins_noisy    = camera_origins    + translation_noise
        ray_origins_noisy       = ray_origins       + translation_noise.unsqueeze(1)
        
        # Apply rotations 
        camera_directions_noisy = th.matmul(rotation_noise, camera_directions.unsqueeze(-1)).squeeze(-1)
        ray_directions_noisy    = th.matmul(rotation_noise.unsqueeze(1), ray_directions.unsqueeze(-1)).squeeze(-1)

        # TODO remove this comment when we have seen the other thing work 
        # camera_origins_noisy = camera_origins + th.randn_like(camera_origins)*translation_noise_sigma
        # camera_directions_noisy = th.matmul(camera_directions.unsqueeze(-1).permute(0,2,1), rotation_noise).squeeze(1)

        return camera_origins_noisy, camera_directions_noisy, ray_origins_noisy, ray_directions_noisy


    # def _get_gaussian_blur_kernel(self, kernel_size: int, relative_sigma: float, max_side_length: int) -> th.Tensor:
    #     # If sigma is 0, return a Dirac delta kernel
    #     if relative_sigma <= sys.float_info.epsilon:
    #         kernel = th.zeros(kernel_size)
    #         kernel[kernel_size//2] = 1
    #     # Else, create 1D Gaussian kernel
    #     # NOTE: Gaussian blur is separable, so 1D kernel can simply be applied twice
    #     else:
    #         kernel = th.linspace(-kernel_size/2, kernel_size/2, kernel_size)
    #         # Calculate inplace exp(-x^2 / (2 * (relative_sigma*max_side_length)^2))
    #         kernel.square_().divide_(-2 * (relative_sigma * max_side_length)**2).exp_()
    #         # Normalize the kernel
    #         kernel.divide_(kernel.sum())


    #     return kernel


    # def _get_blurred_pixel(self, img: th.Tensor, x: int, y: int, gaussian_blur_kernel: th.Tensor):
    #     # NOTE: Assuming x and y are within bounds of img

    #     # Retrive kernel dimensions
    #     kernel_size = gaussian_blur_kernel.shape[0]
    #     kernel_half = kernel_size//2

    #     # Retrieve image dimensions
    #     img_height, img_width = img.shape[:2]

    #     # Calculate padding
    #     left = max(kernel_half - x, 0)
    #     top = max(kernel_half - y, 0)
    #     right = max(kernel_half + x - (img_width-1), 0)
    #     bottom = max(kernel_half + y - (img_height-1), 0)

    #     pad = tv.transforms.Pad(
    #         padding=(left, top, right, bottom), 
    #         padding_mode="reflect"
    #     )

    #     # Pad image and retrieve pixel and neighbors
    #     neighborhood: th.Tensor = pad(img.permute(2, 0, 1))[
    #         :,
    #         (top+y-kernel_half):(top+y+kernel_half)+1, 
    #         (left+x-kernel_half):(left+x+kernel_half)+1,
    #     ].permute(1, 2, 0)


    #     # Blur y-direction and store y-column of pixel
    #     # (H, W, C) -> (W, C)
    #     blurred_y = (neighborhood * gaussian_blur_kernel.view(-1, 1, 1)).sum(dim=0)
    #     # Blur x-direction and store pixel
    #     # (W, C) -> (C)
    #     blurred_pixel = (blurred_y * gaussian_blur_kernel.view(-1, 1)).sum(dim=0)

    #     # Return blurred pixel
    #     return blurred_pixel


    # def gaussian_blur_step(self) -> None:
    #     # Update current variance
    #     self.gaussian_blur_relative_sigma_current *= self.gaussian_blur_relative_sigma_decay
    #     # Get new kernel
    #     self.gaussian_blur_kernel = self._get_gaussian_blur_kernel(
    #         self.gaussian_blur_kernel_size,
    #         self.gaussian_blur_relative_sigma_current,
    #         max(self.image_height, self.image_width)
    #     )


    def subset_dataset(self, image_indices: th.Tensor | list[int|str]):
        """
        Subset data by image indices.

        image_indices can either be image names (str) or image indices (int).

        Creates a copy of the dataset and slices each part of the dataset. 
        This is a shallow copy, however the data is never changed in the dataset,
        So it is completely safe to do this.
        """
        # Create copy 
        output = copy.copy(self)

        if isinstance(image_indices, list):
            # Convert image names to indices
            image_indices = [self.image_name_to_index[idx] if isinstance(idx, str) else int(idx) for idx in image_indices]
        elif isinstance(image_indices, th.Tensor):
            image_indices = image_indices.tolist()
        else:
            raise TypeError(f"image_indices must be either a list or a tensor, but was {type(image_indices)}")

        
        # Slice each part of the dataset
        output.camera_to_worlds = self.camera_to_worlds[image_indices]
        output.camera_origins = self.camera_origins[image_indices]
        output.camera_origins_noisy = self.camera_origins_noisy[image_indices]
        output.ray_directions = self.ray_directions[image_indices]
        output.ray_directions_noisy = self.ray_directions_noisy[image_indices]

        # The index corresponding to each image needs to be keept track of for camera extrinsics 
        output.index_to_index = {i: self.index_to_index[index] for i, index in enumerate(image_indices)}
        output.images = {self.image_index_to_name[i]: self.images[self.image_index_to_name[i]] for i in image_indices}
        output.image_index_to_name = {img_idx_new: self.image_index_to_name[img_idx_old] for img_idx_new, img_idx_old in enumerate(image_indices)}
        output.image_name_to_index = {img_name: img_idx for img_idx, img_name in output.image_index_to_name.items()} # NOTE: using that output.image_index_to_name was created in the line above
        output.n_images = len(output.images)

        return output


    def __getitem__(self, index: int) -> DatasetOutput:
        # Get image index
        img_idx = index // self.image_batch_size

        # Get dataset via image index
        # relic: P, o_r, o_n, d_r, d_n, img = self.dataset[img_idx]
        P = self.camera_to_worlds[img_idx] # (4,4)
        o_r = self.camera_origins[img_idx] # (3, )
        o_n = self.camera_origins_noisy[img_idx] # (3, )
        d_r = self.ray_directions[img_idx] # (H*W, 3)
        d_n = self.ray_directions_noisy[img_idx] # (H*W, 3)
        img = self.images[self.image_index_to_name[img_idx]] # (H, W, 3)
        
        # Get pixel index
        i = index % self.image_batch_size

        return (
            o_r, 
            o_n, 
            d_r.view(-1, 3)[i], 
            d_n.view(-1, 3)[i], 
            # img.view(-1, self.n_sigmas, 3)[i], # TODO the pixel color is now shape = (3,) - should be (N, 3) - where N is the number of gaussian blurred pixel values
            img.view(-1, 3)[i],
            th.tensor(self.index_to_index[img_idx])
        )


    def __len__(self) -> int:
        return self.n_images * self.image_batch_size
    