import json
import os
import pathlib
import math
import sys
from typing import Callable, Optional, cast
import copy
from tqdm import tqdm

import torch as th
from torch.utils.data import Dataset
import torchvision as tv
from PIL import Image, ImageFilter

from model_camera_extrinsics import CameraExtrinsics



# Type alias for dataset output
#  (origin_raw, origin_noisy, direction_raw, direction_noisy, pixel_color_raw, image_index)
DatasetOutput = tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]


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
        verbose: bool=False, 
    ) -> None:
        """Loads images, camera info, and generates rays for each pixel in each image.
        
        Parameters:
        -----------
            * `image_width` `(int)`: Width to resize images to.
            * `image_height` `(int)`: Height to resize images to.
            * `images_path` `(str)`: Path to the image directory.
            * `camera_info_path` `(str)`: Path to the camera info file.
            * `space_transform_scale` `(Optional[float], optional)`:
                Scale parameter for the space transform. Defaults to None, which auto-calculates based on max distance.
                The distances in the original dataset (as read from `camera_info_path`) are devided by this parameter
                to obtain the used dataset. See details below.
            * `space_transform_translate` `(Optional[th.Tensor(3)], optional)`: Translation parameter for the space transform.
                The original camera positions in the dataset are translated by the negated version of this parameter.
                Defaults to `None`, which auto-calculates the mean camera pose and then moves it to the origin.
                See details below.
            * `rotation_noise_sigma` `(float, optional)`: Sigma parameter for the rotation noise in radians. Defaults to 1.0.
            * `translation_noise_sigma` `(float, optional)`: Sigma parameter for the translation noise. Defaults to 1.0.
            * `noise_seed` `(Optional[int], optional)`: Seed for the noise generator. Defaults to None.
            * `verbose` `(bool, optional)`: Whether to print progress. Defaults to False.
        
        Details:
        --------
        `space_transform_scale` and `space_transform_translate` are used to transform the camera to world matrices such that the cameras are centered.
        However, it uses the convention, where the transform arguments are actually the inverse of the transformation that is applied to the camera to world matrices.
        This means, that, camera_poses_original = space_transform_scale * camera_poses_transformed + space_transform_translate,
        where camera_poses_original are the original camera to world matrices, and camera_poses_transformed are the transformed camera to world matrices.

        """
        super().__init__()

        def print_verbose(msg: str) -> None:
            if verbose:
                print(msg)

        # Store image dimensions
        self.image_height, self.image_width = image_height, image_width
        self.image_batch_size = self.image_width * self.image_height

        # Store paths
        self.images_path = images_path
        self.camera_info_path = camera_info_path

        # Store noise parameters
        # NOTE: Rotation is measured in radians
        self.rotation_noise_sigma = rotation_noise_sigma
        self.translation_noise_sigma = translation_noise_sigma
        self.noise_seed = noise_seed


        print_verbose("Loading camera info...")

        # Load camera info
        self.focal_length, self.camera_to_worlds = ImagePoseDataset._load_camera_info(
            self.camera_info_path, 
            self.image_width,
        )

        # Load images
        (
            self.images,
            self.image_name_to_index,
            self.image_index_to_name,
            self.index_to_index,
            self.n_images
        ) = ImagePoseDataset._load_images(
            self.images_path, 
            self.image_height,
            self.image_width, 
            verbose,
        )

        # Transform camera to world matrices
        (
            self.camera_to_worlds, 
            self.space_transform_scale, 
            self.space_transform_translate,
        ) = ImagePoseDataset._transform_camera_to_world(
            self.camera_to_worlds, 
            space_transform_scale,
            space_transform_translate,
            self.image_index_to_name,
            self.n_images,
        )

        print_verbose("Generating rays...")

        # Store the origins of the cameras and the direction their center is facing
        self.camera_origins, self.camera_directions = ImagePoseDataset._get_cam_origs_and_directions(self.camera_to_worlds)

        # Create mesh grid for the directions 
        meshgrid = ImagePoseDataset._get_directions_meshgrid(self.image_height, self.image_width, self.focal_length)
        
        # Transform the unit directions to a direction for each camera
        self.ray_origins, self.ray_directions = ImagePoseDataset._meshgrid_to_world(meshgrid, self.camera_to_worlds)
        
        print_verbose("Applying noise...")

        # Apply noise to get the input data for the model (both rays origins and directions and the centers)
        (
            self.camera_origins_noisy,
            self.camera_directions_noisy,
            self.ray_origins_noisy,
            self.ray_directions_noisy,
        ) = ImagePoseDataset._apply_noise(
            self.camera_origins,
            self.camera_directions,
            self.ray_origins,
            self.ray_directions,
            self.rotation_noise_sigma,
            self.translation_noise_sigma,
            self.noise_seed
        )
        
        print_verbose("Done loading data!")


    ######### static methods ###############

    @staticmethod
    def _load_images(images_path: str,
                     image_height: int,
                     image_width,
                     verbose=False
                     ) -> tuple[th.Tensor, dict[str, int], dict[int, str], dict[int, int], int]: 
        """
        Opens and returns the desired images (with gaussian smoothing applied). 
        Each image is resized, alpha is converted to white, gaussian smoothing is applied,
        channels are permuted and it is converted to a tensor. 

        Parameters:
        -----------
            * `images_path`: str - path to the directory containing the images
            * `img_height`: int - height of the images
            * `img_width`: int - width of the images
        
        Returns:
        --------
            * `images`: `th.Tensor(N, H, W, 3)` - the images, where
                N is the number of images,
                H is the height of the images,
                W is the width of the images, and 
            * `image_name_to_index`: `dict[str, int]` - a dict that maps from image name
                to the index of the corresponding image in images.
            * `image_index_to_name`: `dict[int, str]` - a dict that maps from the index
                of an image in images to the corresponding image name.
            * `index_to_index`: `dict[int, int]` - a dict that maps from the index of an image
                in images to the original index of the image in the original dataset.
                (see Details below)
            * `n_images`: int - the number of images in the dataset.
        
        Details
        -------
        The `index_to_index` dict is used for subsetting the dataset.
        It is a dict that is used to get the index of an image in the origina dataset
        from its index in the current subset of the original dataset:
         * Key: image index in current subset of original dataset, i.e. and integer in [0, n_images_in_current_dataset)
         * Value: the original index of the image, i.e. an integer in [0, n_train_images_originally)

        The reason for this is that we need to be able to remember the original index of the image
        when we subset the dataset, so that we can get the correct camera extrinsics
        I.e this dict is only strictly necassary for training images.
        it is made specifically for the image logger for logging training images.

        """

        image_names_raw = os.listdir(images_path)
        image_names = [pathlib.PurePath(path).stem for path in image_names_raw]
        image_name_to_index = {name: i for i, name in enumerate(image_names)}
        image_index_to_name = {i: name for i, name in enumerate(image_names)}
        n_images = len(image_names)
        index_to_index = {i: i for i in range(n_images)}

        # Load the images as PIL.Image's 
        if verbose: iterator = tqdm(image_names_raw, desc="Loading images")
        else: iterator = image_names_raw
        images = [Image.open(os.path.join(images_path, path)) for path in iterator]

        # Resize each image 
        images = [img.resize((image_width, image_height), Image.BILINEAR) for img in images]

        # Convert alpha to white background
        white_image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 255))
        images = [Image.alpha_composite(white_image, img).convert('RGB') for img in images]

        # Convert to tensor
        PIL_to_tensor = tv.transforms.ToTensor()
        images = th.stack([PIL_to_tensor(img) for img in images]) # shape is (N, 3, H, W)

        # Permute channels to (N, H, W, n_sigmas, C) for each image
        images = images.permute(0, 2, 3, 1)

        return images, image_name_to_index, image_index_to_name, index_to_index, n_images


    @staticmethod
    def _load_camera_info(camera_info_path: str, image_width: int) -> tuple[float, dict[str, th.Tensor]]:
        """
        Loads the camera info from cameras found in the given directory.
        Returns the focal length and a dict that maps from image name to camera to world matrix.

        Parameters:
        -----------
            * `camera_info_path`: `str` - path to the camera info file
            * `image_width`: `int` - width of the images
        
        Returns:
        --------
            * `focal_length`: `float` - the focal length of the camera.\\
                NOTE: Don't know if focal length is the correct word,
                but here it is the distance from the camera to the plane
                that satisfies that the intersections of two neighboring
                camera rays are exactly distance 1 apart.
            * `camera_to_worlds`: `dict[str, th.Tensor(4,4)]` - a dict
                that maps from image name to the camera to world matrix

        """
        def _process_c2w(c2w, path) -> th.Tensor:
            c2w = th.tensor(c2w)
            if not th.allclose(c2w[-1, -1], th.tensor(1.)):
                raise ValueError(f"camera_to_world matrices are expected to have scale 1, found {c2w[-1, -1].item()} in {path}")
            c2w_iden = th.matmul(c2w[:3, :3], c2w[:3, :3].T)
            identity = th.eye(3)

            if not th.allclose(c2w_iden, identity, atol=2e-6, rtol=0.):
                raise ValueError(f"camera_to_world matrices are expected to be orthogonal, found error {(c2w_iden - identity).abs().max()} in {path}") 

            return c2w

 
        # Read info file
        camera_data = json.loads(open(camera_info_path).read())
        
        # Calculate focal length from camera horizontal angle
        focal_length = image_width / 2 / math.tan(camera_data["camera_angle_x"] / 2)
        # Get camera to world matrices
        # NOTE: Projections are scaled to have scale 1
        camera_to_worlds: dict[str, th.Tensor] = {
            pathlib.PurePath(path).stem: _process_c2w(c2w, path)
            for frame in camera_data["frames"] 
            for path, rotation, c2w in [frame.values()] 
        }

        # Return focal length and camera to world matrices
        return focal_length, camera_to_worlds

    @staticmethod
    def _transform_camera_to_world(camera_to_worlds: dict[str, th.Tensor],
                                   space_transform_scale: Optional[float],
                                   space_transform_translate: Optional[th.Tensor],
                                   image_index_to_name: dict[int, str],
                                   n_images: int,
                                   ) -> tuple[th.Tensor, float, th.Tensor]:
    
        """
        Transform the camera to world matrices such that the cameras are centered,
        or according to the given space transform parameters.
        Return the transformed camera to world matrices as one tensor, the scale and translation parameters.

        Parameters:
        ----------
            * `camera_to_worlds`: `dict[str, th.Tensor(4,4)]` - a dict that maps from image name to the camera to world matrix
            * `space_transform_scale`: `Optional[float]` - the scale parameter for the space transform
            * `space_transform_translate`: `Optional[th.Tensor(3)]` - the translation parameter for the space transform
            * `name_to_image_index`: `dict[str, int]` - a dict that maps from image name to the index of the image in the dataset
            * `n_images`: `int` - the number of images in the dataset
        
        Returns:
        --------
            * `camera_to_worlds`: `th.Tensor(N, 4, 4)` - the camera to world matrices, where
                N is the number of images in the dataset
            * `space_transform_scale`: `float` - the scale parameter for the space transform
            * `space_transform_translate`: `th.Tensor(3)` - the translation parameter for the space transform
        """
        # If space transform is not given, initialize transform parameters from data
        # NOTE: Assuming camera_to_world has scale 1

        camera_to_world_raw = th.stack(tuple(camera_to_worlds.values()))
        camera_positions = camera_to_world_raw[:, :3, -1] # (N, 3)
        
        if not th.allclose(camera_to_world_raw[:, -1, -1], th.ones(n_images)):
            raise ValueError("camera_to_worlds matrices are expected to have scale 1")
        
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

        camera_to_worlds_output = th.stack(
            [
                (camera_to_worlds[image_index_to_name[index]] - translate_matrix) / scale_matrix 
                for index in range(n_images)
            ],
            dim = 0
        )

        return (
            camera_to_worlds_output,
            space_transform_scale,
            space_transform_translate,
        )


    @staticmethod
    def _get_cam_origs_and_directions(camera_to_worlds: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """Store the origins of the cameras and the direction their center is facing
        
        Parameters:
        -----------
            * `camera_to_worlds`: `Tensor(N, 4, 4)` - the camera to world matrices, where
                N is the number of images in the dataset
        
        Returns:
        --------
            * `camera_origins`: `Tensor(N, 3)` - the origins of the cameras
            * `camera_directions`: `Tensor(N, 3)` - the direction the cameras are facing - i.e. the direction 
            vector corresponding to the center of the image.
        
        """

        camera_origins = camera_to_worlds[:, :3, 3] # th.stack([c2w[:3, 3] for c2w in camera_to_worlds.values()], dim=0)
        camera_directions = th.matmul(camera_to_worlds[:, :3, :3], th.tensor([0,0,-1.]).view(1,3,1)).squeeze(-1) # th.stack([th.tensor([0., 0., -1.])@c2w[:3, :3].T for c2w in camera_to_worlds.values()], dim=0)# old code: 

        return camera_origins, camera_directions


    @staticmethod
    def _get_directions_meshgrid(image_height: int, image_width: int, focal_length: float) -> th.Tensor:
        """
        Create direction vectors in camera coordinates whose lines intersect with the image plane at the pixel centers.

        Creates a generic meshgrid of directions that can be transformed to be the direction for any
        camera by applying the camera to world matrix.
        Returns the meshgrid flattened to (H*W, 3).

        Parameters:
        -----------
            * `image_height`: `int` - height of the images (H)
            * `image_width`: `int` - width of the images (W)
            * `focal_length`: `float` - the focal length of the camera (see description of focal length in `_load_camera_info()`)
        

        Returns:
        --------
            * `directions`: `th.Tensor(H*W, 3)` - normalised direction vectors flattened to (H*W, 3) - one row for each pixel in the image.

        Details:
        --------
        It uses the convention that the camera is looking in the negative z direction,
        and that direction vectors are normalized (norm 2 = 1).
        Furthermore, the convention is that the top left corner of the image plane is positioned
        at (-image_width/2, image_height/2, -focal_length) in camera coordinates, and
        the bottom right corner is positioned at (image_width/2, -image_height/2, -focal_length).
        The order of the pixels in the meshgrid is by row, meaning that
        the pixel at the i'th row and the j'th column of the image corresponds to the
        (i*image_width + j)'th row of the output of this function. Which matches the order of the pixels in the image.
        

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

    @staticmethod
    def _meshgrid_to_world(meshgrid: th.Tensor, camera_to_worlds: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Transform from the unit meshgrid of directions to a direction for each camera by multiplying with the camera to world matrix.
        Returns the origins and directions of all rays in the dataset.

        Parameters:
        ---------
            * `meshgrid`: `Tensor(H*W, 3)`, output from `_get_directions_meshgrid()`
            * `camera_to_worlds`: `Tensor(N, 4, 4)` - the camera to world matrices, where
                N is the number of images in the dataset
                
        Returns:
        ---------
            * `ray_origins`: `Tensor(N, H*W, 3)`
            * `ray_directions`: `Tensor(N, H*W, 3)`

        Details:
        --------
        NOTE: The final calculation matches the two dimensions that are not shared with an "empty" dimension: 
              camera_to_worlds: (N, 4, 4)   -> unsqueeze(1)                     -> (N, 1, 4, 4)

              meshgrid:         (H*W, 3)    -> unsqueeze(0) and unsqueeze(-1)   -> (1, H*W, 3, 1) (we only use the rotation part of the camera_to_worlds matrix)

              Hence the output is (N, H*W, 3, 1) -> squeeze(-1)                 -> (N, H*W, 3)
        """
        return (
            camera_to_worlds[:, :3, 3].unsqueeze(1).repeat(1, meshgrid.shape[0],1),
            th.matmul(camera_to_worlds[:, :3, :3].unsqueeze(1), meshgrid.unsqueeze(0).unsqueeze(-1)).squeeze(-1) 
        )


    @staticmethod
    def _apply_noise(camera_origins: th.Tensor,
                     camera_directions: th.Tensor,
                     ray_origins: th.Tensor,
                     ray_directions: th.Tensor,
                     rotation_noise_sigma: th.Tensor,
                     translation_noise_sigma: th.Tensor,
                     noise_seed: Optional[int]=None,
                     ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """ 
        Apply noise to the camera origins (a translate) and to the directions (a rotation)

        Returns
        -------
            * `camera_origins_noisy`: `Tensor(N, 3)` - the noisy camera origins
            * `camera_directions_noisy`: `Tensor(N, 3)` - the noisy camera directions
            * `ray_origins_noisy`: `Tensor(N, H*W, 3)` - the noisy ray origins
            * `ray_directions_noisy`: `Tensor(N, H*W, 3)` - the noisy ray directions
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
        translation_noise = th.randn((len(camera_origins), 3), generator=noise_generator)*translation_noise_sigma

        # Apply translations 
        camera_origins_noisy    = camera_origins    + translation_noise
        ray_origins_noisy       = ray_origins       + translation_noise.unsqueeze(1) 
        
        # Apply rotations 
        camera_directions_noisy = th.matmul(rotation_noise, camera_directions.unsqueeze(-1)).squeeze(-1)
        ray_directions_noisy    = th.matmul(rotation_noise.unsqueeze(1), ray_directions.unsqueeze(-1)).squeeze(-1)

        return camera_origins_noisy, camera_directions_noisy, ray_origins_noisy, ray_directions_noisy



    ######### Instance methods ###############
    def subset_dataset(self, image_indices: th.Tensor | list[int|str]):
        """
        Subset data by image indices.

        image_indices can either be image names (str) or image indices (int).

        Creates a copy of the dataset, and then takes out a subset on an image basis. 
        This is a shallow copy, however the data is never changed in the dataset,
        so it is completely safe to do this.

        Parameters:
        ----------
            * `image_indices`: `th.Tensor(N) | list[int|str]` - list of image indices or image names to subset the dataset with.
        
        Returns:
        --------
            * `output`: `ImagePoseDataset` - the subsetted dataset - shallow copy of the original dataset.
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
        output.images = self.images[image_indices]
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
        img = self.images[img_idx] # (H, W, 3)
        
        # Get pixel index
        i = index % self.image_batch_size

        return (
            o_r, 
            o_n, 
            d_r.view(-1, 3)[i], 
            d_n.view(-1, 3)[i], 
            img.view(-1, 3)[i],
            th.tensor(self.index_to_index[img_idx]),
        )


    def __len__(self) -> int:
        return self.n_images * self.image_batch_size