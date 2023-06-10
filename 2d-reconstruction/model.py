import torch as th
import torch.nn as nn
import pytorch_lightning as pl


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
        learning_rate_decay: float = 0.5,
        learning_rate_decay_patience: int = 20,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.width = width
        self.height = height
        
        self.fourier_levels = fourier_levels
        
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_patience = learning_rate_decay_patience
        
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

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)


    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=self.learning_rate_decay, 
            patience=self.learning_rate_decay_patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }
