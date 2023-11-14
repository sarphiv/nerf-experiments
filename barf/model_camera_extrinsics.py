import torch as th
import torch.nn as nn


class CameraExtrinsics(nn.Module):
    def __init__(self, n_train_images: int) -> None:
        super().__init__()
        
        self.size = n_train_images
        # Rotation stored as so3 Lie algebra
        self.rotation = nn.Parameter(th.zeros((n_train_images, 3)))
        self.translation = nn.Parameter(th.zeros((n_train_images, 3)))


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
            -th.eye(3, device=self.rotation.device).view(1, 3, 3), 
            self.rotation.view(-1, 3, 1),
            dim=1
        ))[img_idx]


    def forward_origins(self, i: th.Tensor, o: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forwards pass: Gets the origins in the predicted camera space
        """
        # Translation
        t = self.translation[i]
         
        # Create the rotation matrix
        

        # Get the new rotation and translation
        new_o = o + t

        return new_o, t


    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Forwards pass: Gets the origins and directions in the predicted camera space
        """
        new_o, t = self.forward_origins(i, o)
        R = self.get_rotations(i)
        new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)

        return new_o, new_d, R, t
