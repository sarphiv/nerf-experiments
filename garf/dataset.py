import json
import os
import pathlib
import math
import sys
from typing import Callable, Optional, cast

import torch as th
from torch.utils.data import Dataset
import torchvision as tv


# Type alias for dataset output
#  (origin_raw, origin_noisy, direction_raw, direction_noisy, pixel_color_raw, pixel_color_blur, pixel_relative_blur, image_index)
DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]


class ImagePoseDataset(Dataset[DatasetOutput]):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        images_path: str,
        camera_info_path: str,
        space_transform: Optional[tuple[float, th.Tensor]]=None,
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
            space_transform (Optional[tuple[float, th.Tensor]], optional): Space transform parameters. Defaults to None.
            rotation_noise_sigma (float, optional): Sigma parameter for the rotation noise in radians. Defaults to 1.0.
            translation_noise_sigma (float, optional): Sigma parameter for the translation noise. Defaults to 1.0.
            noise_seed (Optional[int], optional): Seed for the noise generator. Defaults to None.
            gaussian_blur_kernel_size (int, optional): Size of the gaussian blur kernel. Defaults to 5.
            gaussian_blur_relative_sigma_start (float, optional): Initial sigma parameter for the gaussian blur. Set to 0 to disable. Defaults to 0..
            gaussian_blur_relative_sigma_decay (float, optional): Decay factor for the gaussian blur sigma. Defaults to 1..
        """
        super().__init__()
        # Verify parameters
        if gaussian_blur_kernel_size % 2 == 0:
            raise ValueError("Gaussian blur kernel size must be odd.")


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

        # Store gaussian blur parameters
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.gaussian_blur_relative_sigma_start = gaussian_blur_relative_sigma_start
        self.gaussian_blur_relative_sigma_decay = gaussian_blur_relative_sigma_decay
        self.gaussian_blur_relative_sigma_current = self.gaussian_blur_relative_sigma_start



        # Load images
        self.images = self._load_images(
            self.images_path, 
            self.image_width, 
            self.image_height
        )

        # Load camera info
        self.focal_length, self.camera_to_world = self._load_camera_info(
            self.camera_info_path, 
            self.image_width
        )

        # Transform camera to world matrices
        self.camera_to_world, self.space_transform = self._transform_camera_to_world(
            self.camera_to_world, 
            space_transform
        )


        # Get gaussian blur kernel
        self.gaussian_blur_kernel = self._get_gaussian_blur_kernel(
            self.gaussian_blur_kernel_size,
            self.gaussian_blur_relative_sigma_current,
            max(self.image_height, self.image_width)
        )


        # Get raw rays for each pixel in each image
        self.origins_raw, self.directions_raw = self._get_raw_rays(
            self.camera_to_world, 
            self.image_width, 
            self.image_height, 
            self.focal_length
        )
        
        # Get (artificially) noisy rays for each pixel in each image
        self.origins_noisy, self.directions_noisy = self._get_noisy_rays(
            self.origins_raw,
            self.directions_raw,
            self.rotation_noise_sigma,
            self.translation_noise_sigma,
            self.noise_seed
        )


        # Store dataset output
        self.dataset = [
            (
                camera_to_world, 
                self.origins_raw[image_name],
                self.origins_noisy[image_name],
                self.directions_raw[image_name], 
                self.directions_noisy[image_name],
                self.images[image_name]
            ) 
            for image_name, camera_to_world in self.camera_to_world.items()
        ]



    def _load_images(self, image_dir_path, image_width, image_height) -> dict[str, th.Tensor]:
        # Transform from PIL image to Tensor
        transform = cast(
            Callable[[th.Tensor], th.Tensor], 
            tv.transforms.Compose([
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
        camera_to_world: dict[str, th.Tensor] = { 
            pathlib.PurePath(path).stem: th.tensor(camera_to_world) / camera_to_world[-1][-1]
            for frame in camera_data["frames"] 
            for path, rotation, camera_to_world in [frame.values()] 
        }

        # Return focal length and camera to world matrices
        return focal_length, camera_to_world


    def _transform_camera_to_world(self, camera_to_world: dict[str, th.Tensor], space_transform: Optional[tuple[float, th.Tensor]]) -> tuple[dict[str, th.Tensor], tuple[float, th.Tensor]]:
        # If space transform is not given, initialize transform parameters from data
        if space_transform is None:
            camera_positions = th.vstack(tuple(camera_to_world.values()))[:, -1].reshape(-1, 4)
            camera_average_position = camera_positions.mean(dim=0)
            camera_average_position[-1] = 0

            # Get the maximum distance of any two cameras 
            camera_max_distance: float = 3*th.cdist(camera_positions, camera_positions, compute_mode="donot_use_mm_for_euclid_dist").max().item()
            
            # Define space transform 
            space_transform = (camera_max_distance, camera_average_position)


        # Deconstruct the space transform tuple
        (camera_max_distance, camera_average_position) = space_transform

        # Only move the offset
        camera_average_position = th.hstack((th.zeros((4,3)), camera_average_position.unsqueeze(1)))

        # Scale the camera distances
        camera_max_distance_matrix = th.ones((4, 4))
        camera_max_distance_matrix[:-1, -1] = camera_max_distance

        # Move origin to average position of all cameras and scale world coordinates by the 3*the maximum distance of any two cameras
        return (
            { 
                image_name: (camera_to_world - camera_average_position)/camera_max_distance_matrix
                for image_name, camera_to_world 
                in camera_to_world.items() 
            },
            space_transform
        )


    def _get_raw_rays(self, camera_to_world: dict[str, th.Tensor], image_width: int, image_height: int, focal_length: float) -> tuple[dict[str, th.Tensor], dict[str, th.Tensor]]:
        # Create unit directions (H, W, 3) in camera space
        # NOTE: Initially normalized such that z=-1 via the focal length.
        #  Camera is looking in the negative z direction.
        #  y-axis is also flipped.
        y, x = th.meshgrid(
            -th.linspace(-image_height/2, image_height/2, image_height) / focal_length,
            th.linspace(-image_width/2, image_width/2, image_width) / focal_length,
            indexing="ij"
        )
        directions = th.stack((x, y, -th.ones_like(x)), dim=-1)
        directions /= th.norm(directions, p=2, dim=-1, keepdim=True)

        # Return rays keyed by image
        return (
            # Origins: Key by image and get focal points directly from camera to world projection
            {
                image_name: camera_to_world[:3, 3].expand_as(directions)
                for image_name, camera_to_world in camera_to_world.items()
            },
            # Directions: Key by image and get directions directly from camera to world projection
            # Rotate directions (H, W, 3) to world via R (3, 3).
            #  Denote dir (row vector) as one of the directions in the directions tensor.
            #  Then R @ dir.T = (dir @ R.T).T. 
            #  This would yield a column vector as output. 
            #  To get a row vector as output again, simply omit the last transpose.
            #  The inside of the parenthesis on the right side 
            #  is conformant for matrix multiplication with the directions tensor.
            # NOTE: Assuming scale of projection matrix is 1
            { 
                image_name: directions @ camera_to_world[:3, :3].T
                for image_name, camera_to_world in camera_to_world.items()
            }
        )


    def _get_noisy_rays(self, origins: dict[str, th.Tensor], directions: dict[str, th.Tensor], rotation_noise_sigma: Optional[float], translation_noise_sigma: Optional[float], noise_seed: Optional[int]) -> tuple[dict[str, th.Tensor], dict[str, th.Tensor]]:
        # Get amount of cameras
        n_cameras = len(origins)
        
        # Instantiate random number generator
        rng = th.Generator()
        if noise_seed is not None:
            rng.manual_seed(noise_seed)


        # Get random rotation amount in radians
        thetas = th.randn((n_cameras, 1, 1), generator=rng) * rotation_noise_sigma

        # Get random rotation axis
        # NOTE: This is a random point on the unit sphere
        # WARN: Technically a division by zero can happen.
        #  This is however mitigated by applying the Ostrich algorithm :)
        axes = th.randn((n_cameras, 3, 1), generator=rng)
        axes /= th.norm(axes, p=2, dim=1, keepdim=True)
        
        # Get rotation matrices via exponential map from lie algebra so(3) -> SO(3)
        so3 = th.cross(
            -th.eye(3).view(1, 3, 3), 
            thetas * axes,
            dim=1
        )

        rotations = th.matrix_exp(so3)


        # Get random translation amount
        translations = th.randn((n_cameras, 3), generator=rng) * translation_noise_sigma


        # Return rays keyed by image
        return (
            # Origins: Key by image and move focal point
            {
                image_name: origins + trans
                for (image_name, origins), trans in zip(origins.items(), translations)
            },
            # Directions: Key by image and get directions directly from camera to world projection
            # Rotate directions (H, W, 3) via R (3, 3).
            #  Denote dir (row vector) as one of the directions in the directions tensor.
            #  Then R @ dir.T = (dir @ R.T).T. 
            #  This would yield a column vector as output. 
            #  To get a row vector as output again, simply omit the last transpose.
            #  The inside of the parenthesis on the right side 
            #  is conformant for matrix multiplication with the directions tensor.
            { 
                image_name: directions @ rot.T
                for (image_name, directions), rot in zip(directions.items(), rotations)
            }
        )



    def _get_gaussian_blur_kernel(self, kernel_size: int, relative_sigma: float, max_side_length: int) -> th.Tensor:
        # If sigma is 0, return a Dirac delta kernel
        if relative_sigma <= sys.float_info.epsilon:
            kernel = th.zeros(kernel_size)
            kernel[kernel_size//2] = 1
        # Else, create 1D Gaussian kernel
        # NOTE: Gaussian blur is separable, so 1D kernel can simply be applied twice
        else:
            kernel = th.linspace(-kernel_size/2, kernel_size/2, kernel_size)
            # Calculate inplace exp(-x^2 / (2 * (relative_sigma*max_side_length)^2))
            kernel.square_().divide_(-2 * (relative_sigma * max_side_length)**2).exp_()
            # Normalize the kernel
            kernel.divide_(kernel.sum())


        return kernel


    def _get_blurred_pixel(self, img: th.Tensor, x: int, y: int, gaussian_blur_kernel: th.Tensor):
        # NOTE: Assuming x and y are within bounds of img

        # Retrive kernel dimensions
        kernel_size = gaussian_blur_kernel.shape[0]
        kernel_half = kernel_size//2

        # Retrieve image dimensions
        img_height, img_width = img.shape[:2]

        # Calculate padding
        left = max(kernel_half - x, 0)
        top = max(kernel_half - y, 0)
        right = max(kernel_half + x - (img_width-1), 0)
        bottom = max(kernel_half + y - (img_height-1), 0)

        pad = tv.transforms.Pad(
            padding=(left, top, right, bottom), 
            padding_mode="reflect"
        )

        # Pad image and retrieve pixel and neighbors
        neighborhood: th.Tensor = pad(img.permute(2, 0, 1))[
            :,
            (top+y-kernel_half):(top+y+kernel_half)+1, 
            (left+x-kernel_half):(left+x+kernel_half)+1,
        ].permute(1, 2, 0)


        # Blur y-direction and store y-column of pixel
        # (H, W, C) -> (W, C)
        blurred_y = (neighborhood * gaussian_blur_kernel.view(-1, 1, 1)).sum(dim=0)
        # Blur x-direction and store pixel
        # (W, C) -> (C)
        blurred_pixel = (blurred_y * gaussian_blur_kernel.view(-1, 1)).sum(dim=0)

        # Return blurred pixel
        return blurred_pixel


    def gaussian_blur_step(self) -> None:
        # Update current variance
        self.gaussian_blur_relative_sigma_current *= self.gaussian_blur_relative_sigma_decay
        # Get new kernel
        self.gaussian_blur_kernel = self._get_gaussian_blur_kernel(
            self.gaussian_blur_kernel_size,
            self.gaussian_blur_relative_sigma_current,
            max(self.image_height, self.image_width)
        )


    def __getitem__(self, index: int) -> DatasetOutput:
        # Get image index
        img_idx = index // self.image_batch_size

        # Get dataset via image index
        P, o_r, o_n, d_r, d_n, img = self.dataset[img_idx]
        # Get pixel index
        i = index % self.image_batch_size
        
        # Get raw pixel color
        c_r = img.view(-1, 3)[i]

        # If no blur, set color to current pixel
        if self.gaussian_blur_relative_sigma_current <= sys.float_info.epsilon:
            c_b = c_r
        # Else, calculate color via gaussian blur
        else:
            c_b = self._get_blurred_pixel(
                img, 
                i % self.image_width, 
                i // self.image_width, 
                self.gaussian_blur_kernel
            )


        return (
            o_r.view(-1, 3)[i], 
            o_n.view(-1, 3)[i], 
            d_r.view(-1, 3)[i], 
            d_n.view(-1, 3)[i], 
            c_r,
            c_b, 
            th.tensor(self.gaussian_blur_relative_sigma_current), 
            th.tensor(img_idx)
        )


    def __len__(self) -> int:
        return len(self.dataset) * self.image_batch_size