import torch as th
import torch.nn as nn
class CameraExtrinsics(nn.Module):
    def __init__(self, n_train_images: int) -> None:
        super().__init__()
        
        self.size = n_train_images
        # Rotation stored as so3 Lie algebra
        self.rotation = nn.Parameter(th.zeros((n_train_images, 3)))
        self.translation = nn.Parameter(th.zeros((n_train_images, 3)))

    @staticmethod
    def so3_to_SO3(so3: th.Tensor) -> th.Tensor:
        """
        Converts so3 to SO3

        Parameters:
        -----------
            so3: th.Tensor(N, 3) - the so3 Lie algebra to convert, where N is the number of matrices to convert
                Can also be shape (3, ) or (1, 3) (or anything that contains exactly 3 elements).
                so3.view(-1, 3, 1) is called anyway, so the shape is only
                important when input is used to parameterise a batch of matrices
                 - i.e. when more than 3 elements are present in so3
        
        Returns:
        --------
            th.Tensor(N, 3, 3) - the SO3 matrices
        """
        return th.matrix_exp(th.cross(
            -th.eye(3, device=so3.device).view(1, 3, 3), 
            so3.view(-1, 3, 1),
            dim=1
        ))
    

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
        return CameraExtrinsics.so3_to_SO3(self.rotation)[img_idx]


    def forward_origins(self, i: th.Tensor, o: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Forwards pass: Gets the origins in the predicted camera space
        """
        # Translation
        t = self.translation[i]

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
