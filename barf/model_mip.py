
from typing import Literal, Callable, Optional, cast
from itertools import chain
import warnings
import math

import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import nerfacc

from model_interpolation_architecture import NerfModel, PositionalEncoding
from model_interpolation import NerfInterpolation, uniform_sampling_strategies, integration_strategies
from model_camera_calibration import CameraCalibrationModel
from data_module import ImagePoseDataModule

# TODO Fix such that it always uses the same model as both proposal and radiance
class MipNeRF(NerfInterpolation):

    def __init__(
        self, 
        near_sphere_normalized: float,
        far_sphere_normalized: float,
        model_radiance: NerfModel,
        samples_per_ray_radiance: int,
        uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
        uniform_sampling_offset_size: float = 0.,
        integration_strategy: integration_strategies = "middle",
        samples_per_ray_proposal: int = 0,
        ):
        NerfInterpolation.__init__(self,
            self, 
            near_sphere_normalized=near_sphere_normalized,
            far_sphere_normalized=far_sphere_normalized,
            model_radiance=model_radiance,
            model_proposal=model_radiance if samples_per_ray_proposal > 0  else None,
            samples_per_ray_radiance=samples_per_ray_radiance,
            uniform_sampling_strategy=uniform_sampling_strategy,
            uniform_sampling_offset_size=uniform_sampling_offset_size,
            integration_strategy=integration_strategy,
            samples_per_ray_proposal=samples_per_ray_proposal,
        )

        self.param_groups = self.model_radiance.param_groups

    def _step_helper(self, batch, batch_idx, purpose: Literal["train", "val"]):

        # unpack batch
        (
            ray_origs_raw, 
            ray_origs_pred, 
            ray_dirs_raw, 
            ray_dirs_pred, 
            ray_colors_raw, 
            img_idx,
            pixel_width
        ) = batch

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, pixel_width)


        # compute the loss
        loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:,0])
        psnr = -10 * math.log10(float(loss.detach().item()))
        # Log metrics
        logs  = {f"{purpose}_loss_fine": loss,
                    f"{purpose}_psnr": psnr,
                    }

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:,0])
            loss = loss + loss_coarse*0.1
            logs[f"{purpose}_loss_coarse"] = loss_coarse

        self.log_dict(logs)

        if loss.isnan():
            loss = th.tensor(1., requires_grad=True)
            print("loss was nan - no optimization step performed")

        return loss


class MipBarf(CameraCalibrationModel):
    
    def __init__(self,
            model_radiance: NerfModel,
            samples_per_ray_radiance: int,
            n_training_images: int,
            camera_learning_rate_start: float,
            camera_learning_rate_stop: float,
            camera_learning_rate_decay_end: int = -1,
            near_sphere_normalized: float = 2.,
            far_sphere_normalized: float = 8.,
            uniform_sampling_strategy: uniform_sampling_strategies = "stratified_uniform",
            uniform_sampling_offset_size: float = 0.,
            samples_per_ray_proposal: int = 0,
            sigma_decay_start_step: int = 0,
            sigma_decay_end_step: int = 0,
            start_blur_sigma: float = 0.,
            start_pixel_width_sigma: float = 0.0,
            ):
        """
        mip barf model
            * sigma_decay_start_step: int - When to start decaying sigma (optimization step)
            * sigma_decay_end_step: int - when to stop decaying sigma
            * start_blur_sigma: float - the starting sigma blur value
            * start_pixel_width_sigma: float - the starting sigma pixel width value (for cone casting)
        
        Details:
        ----------
        the sigma follows the following schedule (both blur and pixel width):

            * current step < start step: sigma = start sigma
            * start step < current step < end step: sigma follows exponential decay from start sigma at start step down to 1/4 at stop step
            * end step < current step: sigma = 0.



        For the following arguments, see NerfInterpolation and CameraCalibrationModel:
            * model_radiance
            * samples_per_ray_radiance
            * n_training_images
            * camera_learning_rate_start
            * camera_learning_rate_stop
            * camera_learning_rate_decay_end
            * max_gaussian_sigma [not used]
            * near_sphere_normalized
            * far_sphere_normalized
            * uniform_sampling_strategy
            * uniform_sampling_offset_size
            * samples_per_ray_proposal
        """
        
        CameraCalibrationModel.__init__(self,
            model_radiance=model_radiance,
            model_proposal=model_radiance if samples_per_ray_proposal > 0 else None,
            samples_per_ray_radiance=samples_per_ray_radiance,
            n_training_images=n_training_images,
            camera_learning_rate_start=camera_learning_rate_start,
            camera_learning_rate_stop=camera_learning_rate_stop,
            camera_learning_rate_decay_end=camera_learning_rate_decay_end,
            max_gaussian_sigma=None,
            near_sphere_normalized=near_sphere_normalized,
            far_sphere_normalized=far_sphere_normalized,
            uniform_sampling_strategy=uniform_sampling_strategy,
            uniform_sampling_offset_size=uniform_sampling_offset_size,
            integration_strategy="middle",
            samples_per_ray_proposal=samples_per_ray_proposal,
            )
        


        self.start_blur_sigma = float(start_blur_sigma)
        self.start_pixel_width_sigma = float(start_pixel_width_sigma)
        
        self.sigma_decay_start_step = sigma_decay_start_step
        self.sigma_decay_end_step   = sigma_decay_end_step

        self.sigma_schedule = 1.



        self.param_groups = [param_group for model in [self.model_radiance, self.camera_extrinsics]
                             for param_group in model.param_groups]

        self.model_radiance.position_encoder.pixel_width_sigma = self.start_pixel_width_sigma

    def update_sigma_schedule(self, current_step):
        """
        update the sigma schedule

        the sigma schedule follows the following schedule:

            * current step < start step: sigma = start sigma
            * start step < current step < end step: sigma follows exponential decay
            from start sigma at start step down to 1/4 at end step
            * end step < current step: sigma = 0.

        """


        if current_step < self.sigma_decay_start_step:
            sigma_schedule = 1.

        elif self.sigma_decay_start_step <= current_step <= self.sigma_decay_end_step:
            sigma_schedule = (
                (
                    0.25/(max(self.start_blur_sigma,self.start_pixel_width_sigma))
                )**(
                    (
                        self.sigma_decay_start_step - current_step
                    )/(
                        self.sigma_decay_start_step - self.sigma_decay_end_step
                    )
                )

            )

        else:
            sigma_schedule = 0.
        
        self.sigma_schedule = sigma_schedule


    @property
    def current_blur_sigma(self):
        """
        the current value of sigma to use for the gaussian blur

        if blur_follows_sigma is True, then the image is blurred with the
        value of sigma_schedule - otherwise not blurred
        """

        sigma = self.sigma_schedule*self.start_blur_sigma
        if sigma < 0.25: return 0.0
        else: return sigma


    @property
    def current_pixel_width_sigma(self):
        sigma = self.sigma_schedule*self.start_pixel_width_sigma
        if sigma < 0.25: return 0.0
        else: return sigma

    # @property
    # def pixel_width_scalar(self):
    #     """
    #     DEPRECATED - just returns 1 - replaced by self.model_radiance.position_encoder.gaussian_blur_sigma
    #     the correction factor to multiply the pixel width with in order to
    #     account for the variance introduced by the gaussian blur
    #     in the mip nerf model

    #     if pixel_width_follows_sigma is 1, then the pixel width passed to the
    #     positional encoding is corrected - otherwise not
        
    #     """
    #     return 1.
    #     if self.pixel_width_follows_sigma:
    #         return (12*self.sigma_schedule**2 + 1)**0.5
    #     else:
    #         return 1.

    def _step_helper(self, batch, batch_idx, purpose: Literal['train', 'val']):

        if purpose == "train":
            n_batches = len(self.trainer.train_dataloader)
            current_step = self.trainer.current_epoch*n_batches + batch_idx
            self.update_sigma_schedule(current_step)

            self.model_radiance.position_encoder.pixel_width_sigma = self.current_pixel_width_sigma

            batch = self.training_transform(batch)
        elif purpose == "val":
            batch = self.validation_transform(batch)
        else:
            raise ValueError(f"purpose={purpose} is invalid")
        
        # Unpack batch

        # blur the image
        (
            ray_origs_raw,
            ray_origs_pred,
            ray_dirs_raw,
            ray_dirs_pred,
            ray_colors_raw,
            img_idx,
            pixel_width
        ) = cast(ImagePoseDataModule, self.trainer.datamodule).get_blurred_pixel_colors(batch, self.current_blur_sigma)

        # compute the pose error

        # Forward pass
        ray_colors_pred_fine, ray_colors_pred_coarse = self.forward(ray_origs_pred, ray_dirs_pred, pixel_width)


        # compute the loss
        loss = nn.functional.mse_loss(ray_colors_pred_fine, ray_colors_raw[:,0])
        psnr = -10 * math.log10(float(loss.detach().item()))
        # Log metrics
        logs  = {f"{purpose}_loss_fine": loss,
                f"{purpose}_psnr": psnr,
                # f"pixel_width_scalar": self.pixel_width_scalar,
                f"PE_sigma": self.model_radiance.position_encoder.pixel_width_sigma,
                f"blur_sigma": self.current_blur_sigma,
                    }

        if self.proposal:
            loss_coarse = nn.functional.mse_loss(ray_colors_pred_coarse, ray_colors_raw[:,0])
            loss = loss + loss_coarse*0.1
            logs[f"{purpose}_loss_coarse"] = loss_coarse

        if ((purpose == "train" and batch_idx % 100 == 0) or (purpose == "val" and batch_idx == 0)):
            pose_error = CameraCalibrationModel.compute_pose_error(self) 
            logs["pose_errors"] = pose_error
        self.log_dict(logs)

        if loss.isnan():
            loss = th.tensor(1., requires_grad=True)
            print("loss was nan - no optimization step performed")

        return loss

# if __name__ == "__main__":
    #### test pose error

    # class DummyTrainer:
    #     def __init__(self, datamodule):
    #         self.datamodule = datamodule

    # from positional_encodings import IntegratedFourierFeatures, FourierFeatures
    # import matplotlib.pyplot as plt
    # from model_camera_extrinsics import CameraExtrinsics
    # position_encoding = IntegratedFourierFeatures(10)
    # direction_encoding = FourierFeatures(4)
    # mip_radiance = NerfModel(4, 256, True, False, 2, position_encoding, direction_encoding)
    # model = MipBarf(mip_radiance, 256, 20, 100, 0, 0, -1, samples_per_ray_proposal=20,
    #                 gaussian_sigma_decay_start_step=20, gaussian_sigma_decay_end_step=60,
    #                 pixel_width_follows_sigma=1, blur_follows_sigma=1)
    

    # for translation_noise_sigma in [0, 0.6, 4, 30]:

    #     dm = ImagePoseDataModule(
    #         image_width=10,
    #         image_height=10,
    #         space_transform_scale=1.,
    #         space_transform_translate=th.Tensor([0,0,0]),
    #         scene_path="../data/lego",
    #         verbose=False,
    #         validation_fraction=0.02,
    #         validation_fraction_shuffle=1234,
    #         gaussian_blur_sigmas = [0],
    #         rotation_noise_sigma = 0,
    #         translation_noise_sigma = translation_noise_sigma,
    #         batch_size=1024,
    #         num_workers=5,
    #         shuffle=True,
    #         pin_memory=True,
    #     )
    #     dm.setup("fit")
    #     trainer = DummyTrainer(dm)
    #     camera_origins_raw_original = dm.dataset_train.camera_origins
    #     camera_origins_noisy_original = dm.dataset_train.camera_origins_noisy
    #     model.trainer = trainer

    #     errors_transformed = []
    #     errors_no_tranform = []
    #     error_true = (((camera_origins_raw_original - camera_origins_noisy_original)**2).sum(dim=1)**0.5).mean()



    #     for i in range(1000):
    #         random_rotation = CameraExtrinsics.so3_to_SO3(th.randn(1,3)*12)
    #         random_translation = th.randn(1,3)*20 + 2.5
    #         random_scale = th.rand(1)*4+0.1


    #         dm.dataset_train.camera_origins = camera_origins_raw_original
    #         dm.dataset_train.camera_origins_noisy = camera_origins_noisy_original
    #         error = model.compute_pose_error()
    #         errors_no_tranform.append(error.detach().item())

    #         dm.dataset_train.camera_origins = th.matmul(random_rotation, camera_origins_raw_original.unsqueeze(-1)).squeeze() + random_translation
    #         dm.dataset_train.camera_origins_noisy = camera_origins_noisy_original*1/random_scale
    #         error = model.compute_pose_error()
    #         errors_transformed.append(error.detach().item())

    #     errors_transformed = th.tensor(errors_transformed)
    #     errors_no_tranform = th.tensor(errors_no_tranform)
    #     print(translation_noise_sigma,error_true.item())
    #     print(errors_transformed.mean().item(), errors_transformed.max().item(), errors_transformed.min().item(), errors_transformed.std().item())
    #     print(errors_no_tranform.mean().item(), errors_no_tranform.max().item(), errors_no_tranform.min().item(), errors_no_tranform.std().item())



        





    



    ####### testing sigma schedule and gaussian blur
    # from positional_encodings import IntegratedFourierFeatures, FourierFeatures
    # import matplotlib.pyplot as plt
    # dm = ImagePoseDataModule(
    #     image_width=200,
    #     image_height=200,
    #     space_transform_scale=1.,
    #     space_transform_translate=th.Tensor([0,0,0]),
    #     scene_path="../data/lego",
    #     verbose=True,
    #     validation_fraction=0.02,
    #     validation_fraction_shuffle=1234,
    #     gaussian_blur_sigmas = [20,13,10,7,5,3,2,0],
    #     rotation_noise_sigma = 0,
    #     translation_noise_sigma = 0,
    #     batch_size=1024,
    #     num_workers=5,
    #     shuffle=True,
    #     pin_memory=True,
    # )
    # dm.setup("fit")
    # position_encoding = IntegratedFourierFeatures(10)
    # direction_encoding = FourierFeatures(4)
    # mip_radiance = NerfModel(4, 256, True, False, 2, position_encoding, direction_encoding)

    # steps = th.linspace(0, 100, 101)
    # for blur_follows_sigma in [False, True]:
    #     for pixel_width_follows_sigma in [0, 1]:

    #         model = MipBarf(mip_radiance, 256, 20, 100, 0, 0, -1, samples_per_ray_proposal=20,
    #                         gaussian_sigma_decay_start_step=20, gaussian_sigma_decay_end_step=60,
    #                         pixel_width_follows_sigma=pixel_width_follows_sigma, blur_follows_sigma=blur_follows_sigma)
            
    #         sigma_schedules = []
    #         gaussian_sigmas = []
    #         pixel_widths = []
    #         for step in steps:
    #             model.update_sigma_schedule(step.item())
    #             sigma_schedules.append(model.sigma_schedule)
    #             gaussian_sigmas.append(model.current_gaussian_sigma)
    #             pixel_widths.append(model.pixel_width_scalar)
    #         fig = plt.figure()
    #         fig.suptitle(f"blur_follows_sigma={blur_follows_sigma}, pixel_width_follows_sigma={pixel_width_follows_sigma}")
    #         fig.add_subplot(1, 3, 1)
    #         plt.plot(steps, sigma_schedules, label="sigma schedule")
    #         plt.legend()
    #         fig.add_subplot(1, 3, 2)
    #         plt.plot(steps, gaussian_sigmas, label="gaussian sigma")
    #         plt.legend()
    #         fig.add_subplot(1, 3, 3)
    #         plt.plot(steps, pixel_widths, label="pixel width scalar")
    #         plt.legend()

    #         plt.savefig(f"sigma_schedule_blur_follows_sigma={blur_follows_sigma}_pixel_width_follows_sigma={pixel_width_follows_sigma}.png")
    #         plt.cla()

    #         dataloader = dm.val_dataloader()

    #         output = []

    #         for step in [0, 10, 20, 30, 40, 50, 60]:
    #             model.update_sigma_schedule(step)
    #             output = []

    #             for batch in dataloader:

                    

    #                 # blur the image
    #                 (
    #                     ray_origs_raw,
    #                     ray_origs_pred,
    #                     ray_dirs_raw,
    #                     ray_dirs_pred,
    #                     ray_colors_raw,
    #                     img_idx,
    #                     pixel_width
    #                 ) = dm.get_blurred_pixel_colors(batch, model.current_gaussian_sigma)

    #                 output.append(ray_colors_raw[:, 0])
                
    #             output = th.cat(output, dim=0)

    #             output = output.view(-1, 200, 200, 3)

    #             plt.imsave(f"step={step}_blurred_image_sigma_schedule={model.sigma_schedule}_blur_follows_sigma={blur_follows_sigma}_pixel_width_follows_sigma={pixel_width_follows_sigma}.png", output[0].detach().cpu().numpy())





