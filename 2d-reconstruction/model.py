import torch as th
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt


class FourierFeatures(nn.Module):
    def __init__(self, levels: int):
        super().__init__()

        self.levels = levels


    def forward(self, x: th.Tensor):
        scale = (2**th.arange(self.levels) * th.pi) \
            .repeat(x.shape[1]) \
            .to(x.device)
        args = x.repeat_interleave(self.levels, dim=1) * scale

        # NOTE: Sines and cosines on the same level are not adjacent 
        #  as in the original paper. Network should be invariant to this,
        #  so there should be no loss difference. Computation is faster though.
        return th.hstack((th.cos(args), th.sin(args)))


class Nerf2d(pl.LightningModule):
    def __init__(
        self, 
        width: int, 
        height: int, 
        fourier_levels: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        
        self.width = width
        self.height = height
        
        self.fourier_levels = fourier_levels
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        
        self.model = nn.Sequential(
            FourierFeatures(levels=self.fourier_levels),
            nn.Linear(2*2*self.fourier_levels, 256),
            
            # nn.Linear(2, 256),
            
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3),
            nn.Sigmoid()
        )


    def forward(self, x: th.Tensor):
        return self.model(x)


    def training_step(self, batch: tuple[th.Tensor, th.Tensor], batch_idx: int):
        x, y = batch

        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)


        # Draw training image
        # plt.scatter(x[:, 1].cpu().numpy(), x[:, 0].cpu().numpy(), c=y.cpu().numpy(), s=0.2)
        # plt.draw()
        # plt.pause(0.005)
        

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)


        if np.random.rand() > 0.99:
            x, y = th.meshgrid(
                th.linspace(0, 1, self.width, device=self.device), 
                th.linspace(0, 1, self.height, device=self.device),
                indexing="ij"
            )
            location = th.hstack((
                x.flatten().unsqueeze(1),
                y.flatten().unsqueeze(1)
            ))
            
            print(loss.item())
            rgb = self(location).view(self.width, self.height, 3).permute(1, 0, 2).to("cpu").detach().numpy()
            rgb = np.array(rgb, dtype=np.float32)

            plt.imshow(rgb)
            plt.draw()
            plt.pause(0.5)


    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="max", factor=0.5, patience=1
        # )

        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "train_loss"
        }
