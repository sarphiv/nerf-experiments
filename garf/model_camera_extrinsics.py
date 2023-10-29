import torch as th
import torch.nn as nn


class CameraExtrinsics(nn.Module):
    def __init__(self, rotation_init_sigma: float, translation_init_sigma: float, n_train_images: int) -> None:
        super().__init__()
        
        self.size = n_train_images
        # Rotation stored as so3 Lie algebra
        self.rotation = nn.Parameter(th.randn((n_train_images, 3))*rotation_init_sigma)
        self.translation = nn.Parameter(th.randn((n_train_images, 3))*translation_init_sigma)


    def get_rotations(self, img_idx: th.Tensor) -> th.Tensor:
        """
        Get the rotation matrices for the given indexs

        Parameters:
        -----------
            img_idx: th.Tensor - the indices of the associated rotation matrices for each image to get

        Returns:
        --------
            th.Tensor - the rotation matrices
        """
        return th.matrix_exp(th.cross(
            -th.eye(3).view(1, 3, 3), 
            self.rotation[img_idx].view(-1, 3, 1),
            dim=1
        ))


    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forwards pass: Gets the origins and directions in the predicted camera space
        """
        # Translation
        t = self.translation[i]
         
        # Create the rotation matrix
        R = self.get_rotations(i)

        # Get the new rotation and translation
        new_o = th.matmul(R, o.unsqueeze(-1)).squeeze(-1) + t
        new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)

        return new_o, new_d, R, t
