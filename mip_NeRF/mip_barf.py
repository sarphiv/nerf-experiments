from typing import cast
from tqdm import tqdm

from typing import Optional
import torch as th
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mip_model import MipNerf, MipNerfModel, NerfInterpolation
from data_module import DatasetOutput, ImagePoseDataset

class CameraExtrinsics(nn.Module):
    def __init__(self, noise_rotation, noise_translation, size, device, global_init_rot: th.Tensor | None = None, global_init_trans: th.Tensor | None = None) -> None:
        super().__init__()
        
        self.device = device
        self.size = size    # The amount of images

        p = th.randn((size, 6))*th.tensor([noise_rotation]*3 + [noise_translation]*3)
        if global_init_rot is not None and global_init_trans is not None:
            p[:, 3:] = th.matmul(p[:, :3], global_init_rot) + global_init_trans

        self.params = nn.Parameter(p)   # a, b, c, tx, ty, tz

        # self.register_buffer("params", th.zeros((size, 6), requires_grad=False))   # a, b, c, tx, ty, tz
        self.register_buffer("a_help_mat", th.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], requires_grad=False))
        self.register_buffer("b_help_mat", th.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], requires_grad=False))
        self.register_buffer("c_help_mat", th.tensor([[0, 0, 0], [0, 0, 1], [0, -1, 0]], requires_grad=False))


    def get_R(self, i: th.Tensor):
        # Find the appropriate parameters for the index
        # Rotation
        a = self.params[:, 0].view(-1, 1, 1)*self.a_help_mat
        b = self.params[:, 1].view(-1, 1, 1)*self.b_help_mat
        c = self.params[:, 2].view(-1, 1, 1)*self.c_help_mat
         
        # Create the rotation matrix
        R = th.matrix_exp(a+b+c)

        R = R[i]

        return R
    
    def get_t(self, i: th.Tensor | None = None):

        if i is None: i = th.arange(self.size)
        # Translation
        trans = self.params[i, 3:] 

        return trans



    def forward(self, i: th.Tensor, o: th.Tensor, d: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        i = i.squeeze(-1)
        R = self.get_R(i)
        trans = self.get_t(i)

        new_o = th.matmul(R, o.unsqueeze(-1)).squeeze(-1) + trans
        new_d = th.matmul(R, d.unsqueeze(-1)).squeeze(-1)

        return new_o, new_d, trans, R


class MipBARFModel(MipNerfModel):
    def __init__(self, n_hidden: int,
                 hidden_dim: int,
                 fourier: tuple[bool, int, int],
                 n_segments: int,
                 distribute_variance: bool | None = False,
                 camera_extrinsics: CameraExtrinsics | None = None):
        super().__init__(n_hidden, hidden_dim, fourier, n_segments, distribute_variance)
        
        self.camera_extrinsics = camera_extrinsics
        self.camera_extrinsics


    def forward(self, pos: th.Tensor, dir: th.Tensor, t_start: th.Tensor, t_end: th.Tensor, pixel_width: th.Tensor, cam_idx: th.Tensor) -> tuple[th.Tensor, th.Tensor]:

        if not (self.camera_extrinsics is None) and not (cam_idx is None):
            new_pos, new_dir, _ ,_ = self.camera_extrinsics(cam_idx, pos, dir)
        else:
            new_pos = pos
            new_dir = dir

        return super().forward(new_pos, new_dir, t_start, t_end, pixel_width, cam_idx)


class MipBARF(MipNerf):
    def __init__(self,
                 near_sphere_normalized: float,
                 far_sphere_normalized: float,
                 samples_per_ray: int,
                 n_hidden: int,
                 proposal: tuple[bool, int],
                 fourier: tuple[bool, int, int],
                 n_segments: int,
                 learning_rate: float = 0.0001,
                 learning_rate_decay: float = 0.5,
                 weight_decay: float = 0,
                 distribute_variance: bool | None = False,
                 camera_extrinsics: CameraExtrinsics | None = None
                 ):
        
        # super(pl.LightningModule, self).__init__()
        super(NerfInterpolation, self).__init__()
        self.save_hyperparameters()

        # Hyper parameters for the optimizer 
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.weight_decay = weight_decay

        # The near and far sphere distances 
        self.near_sphere_normalized = near_sphere_normalized
        self.far_sphere_normalized = far_sphere_normalized
        

        # If there is a proposal network separate the samples into coarse and fine sampling
        self.proposal = proposal[0]
        if self.proposal:
            self.samples_per_ray_coarse = proposal[1]
            self.samples_per_ray_fine = samples_per_ray - proposal[1]

        else: 
            self.samples_per_ray_coarse = samples_per_ray
            self.samples_per_ray_fine = 0
        
        # build the model(s)
        self.model_fine = MipBARFModel(
            n_hidden,
            hidden_dim=256,
            fourier=fourier,
            n_segments=n_segments,
            distribute_variance=distribute_variance,
            camera_extrinsics=camera_extrinsics
        )

        self.model_coarse = self.model_fine

        self.camera_extrinsics = camera_extrinsics
        self.cam_pose_transform_params = None


    def kabsch_algorithm(self, point_cloud_from: th.Tensor, point_cloud_to: th.Tensor): 
        """
        Align "point_cloud_from" to "point_cloud_to" with a matrix R and a vector t.
        align_rotation: helper function that optimizes the subproblem ||P - Q@R||^2 using SVD
        
        Parameters:
        -----------
            point_cloud_from: th.Tensor - the point cloud where we transform from
            point_cloud_to: th.Tensor - the point cloud where we transform to

        Returns:
        --------
            R: th.Tensor - the rotation matrix
            t: th.Tensor - the translation vector
        """
        def align_rotation(P: th.Tensor, Q: th.Tensor):
            """
            Optimize ||P - Q@R||^2 using SVD
            """
            H = P.T@Q
            U, S, V = th.linalg.svd(H.to(dtype=th.float)) # Conventional notation is: U, S, V^T 
            d = th.linalg.det((V.T@U.T).to(dtype=th.float))
            K = th.eye(len(S), dtype=th.float, device=P.device)
            K[-1,-1] = d
            R = V.T@K@U.T
            return R.to(dtype=P.dtype)

        # TODO: Seems like a scaling operation is missing here

        # Translate both point clouds to the origin 
        mean_from = th.mean(point_cloud_from, dim=0, keepdim=True)
        mean_to = th.mean(point_cloud_to, dim=0, keepdim=True)

        centered_from = point_cloud_from - mean_from
        centered_to = point_cloud_to - mean_to

        scale_from = (centered_from**2).sum()**0.5
        scale_to = (centered_to**2).sum()**0.5
        scale = scale_to/scale_from

        # Find the rotation matrix such that ||(C1 - m1) - (C2 - m2)@R||^2 is minimized
        R = align_rotation(centered_to, centered_from)

        # The translation vector is now 
        t = mean_to - mean_from@R

        estimated_to = scale_to/scale_from*centered_from@R + mean_to

        return R, t, scale
    
    def update_cam_pose_transform_params(self):
        if self.camera_extrinsics is not None:
            point_cloud_from = th.stack([M[0][:3,3] for M in self.trainer.datamodule.dataset_train.dataset]).to(self.device) # type: ignore
            point_cloud_to = self.camera_extrinsics.get_t() + point_cloud_from
            self.cam_pose_transform_params = self.kabsch_algorithm(point_cloud_from, point_cloud_to)


    
    def transform_batch_cam_poses(self, batch: DatasetOutput):

        if self.camera_extrinsics is not None:
            if self.cam_pose_transform_params is None:
                raise ValueError

            R,t,c = self.cam_pose_transform_params

            cam_to_world, ray_origs, ray_dirs, ray_colors, pixel_width, camera_index = batch
            ray_origs_mean = ray_origs.mean(dim=0, keepdim=True)
            ray_origs_scaled = (ray_origs - ray_origs_mean) * c + ray_origs_mean
            ray_origs_transformed = ray_origs_scaled@R + t

            ray_dirs_transformed = ray_dirs@R

            batch = cam_to_world, ray_origs_transformed, ray_dirs_transformed, ray_colors, pixel_width, camera_index

        return batch


    def render_image(self, name):

        dataset = cast(ImagePoseDataset, self.trainer.datamodule.dataset_val) # type: ignore
        

        data_loader = DataLoader(
            dataset=TensorDataset(
                dataset.origins[name].view(-1, 3), 
                dataset.directions[name].view(-1, 3)
            ),
            batch_size=1024,
            num_workers=4,
            shuffle=False
        )

        pixel_width = th.tensor(dataset.pixel_width, dtype=dataset.origins[name].dtype, device=self.device)
        pixel_width = pixel_width.view(1,1).expand(1024, 1)

        # Iterate over batches of rays to get RGB values
        rgb = th.empty((dataset.image_batch_size, 3), dtype=cast(th.dtype, self.dtype))
        i = 0
        if self.cam_pose_transform_params is not None:
            R,t,c = self.cam_pose_transform_params

        for ray_origs, ray_dirs in tqdm(data_loader, desc="Predicting RGB values", leave=False):
            # Prepare for model prediction
            ray_origs = cast(th.Tensor,ray_origs).to(self.device)
            ray_dirs = cast(th.Tensor,ray_dirs).to(self.device)
            if self.cam_pose_transform_params is not None:
                ray_origs_mean = ray_origs.mean(dim=0, keepdim=True)
                ray_origs_scaled = (ray_origs - ray_origs_mean) * c + ray_origs_mean
                ray_origs = ray_origs_scaled@R + t

                ray_dirs = ray_dirs@R

            
            # Get size of batch
            batch_size = ray_origs.shape[0]
            
            # Predict RGB values
            rgb[i:i+batch_size, :] = self.forward(ray_origs, ray_dirs, pixel_width[:batch_size], None)[0].clip(0, 1).cpu()
        
            # Update write head
            i += batch_size
        
        image = rgb.view(dataset.image_height, dataset.image_width, 3).numpy()
        return image

    def validation_step(self, batch: DatasetOutput, batch_idx: int):

        self.update_cam_pose_transform_params()
        batch = self.transform_batch_cam_poses(batch)

            # # Get the raw and noise origins 
            # (origs_raw, _), (origs_noisy, _) = cast(ImagePoseDataModule, self.trainer.datamodule).train_camera_center_rays(self.device) # type: ignore
            # img_idxs = th.arange(len(origs_raw), device=origs_raw.device)

            # # Get the predicted origins 
            # origs_pred, _, _ = self.camera_extrinsics.forward_origins(img_idxs, origs_noisy)

            # # Align raw space to predicted model space
            # post_transform_params = self.kabsch_algorithm(origs_raw, origs_pred)


        return super().validation_step(batch, batch_idx)
    


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    point_cloud_from = th.tensor([[1,5.],
                                  [2,8],
                                  [1,1],
                                  [8,10],
                                  [-1,1],
                                  [3,6]
                                  ])

    R = th.matrix_exp(th.tensor([[0.,-1],
                                [1,0]
                                ]))
    
    t = th.tensor([[2,4]])

    c = 2.

    point_cloud_from_mean = point_cloud_from.mean(dim=0, keepdim=True)
    point_cloud_to = ((point_cloud_from - point_cloud_from_mean)*c + point_cloud_from_mean)@R + t     

    Rhat, that, chat = MipBARF.kabsch_algorithm(None, point_cloud_from, point_cloud_to) #type: ignore

    point_cloud_to_hat = th.matmul((point_cloud_from - point_cloud_from_mean)*chat + point_cloud_from_mean, Rhat) + that
    # point_cloud_to_hat = ((point_cloud_from - point_cloud_from_mean)*chat + point_cloud_from_mean)@Rhat + that

    print(Rhat,R,sep="\n")
    print(chat,c,sep="\n")
    print(that,t,sep="\n")

    print(point_cloud_to - point_cloud_to_hat)