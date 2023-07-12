import torch as th
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any, Callable, Iterator, Literal, cast

from data_loader import SingleImageDataModule
import math

import numpy as np
import os


class INGPTable(nn.Module):
    def __init__(self, resolution, table_size, n_features, pi1, pi2):
        super().__init__()
        self.resolution = resolution
        self.table_size = table_size
        self.n_features = n_features
        self.pi1 = pi1
        self.pi2 = pi2

        self.bijective = table_size >= (resolution + 1)**2

        if self.bijective:
            self.table = nn.Parameter(
                (th.rand(((resolution+1)**2, n_features))*2 - 1)*10**(-4)
                )
        else:
            self.table = nn.Parameter(
                (th.rand((table_size, n_features))*2 - 1)*10**(-4)
                )
    
    def compute_idx(self, corners):
        if self.bijective:
            idx = th.sum(corners * th.tensor([1, self.resolution+1], device=corners.device), dim=2)
        else:
            idx = self.hash(corners)
        return idx

    def hash(self, x):
        # x: (batch_size, 2**d, d) - d=3 (we are gonna have d 2's)
        # output: (batch_size, 2**d)

        y1 = self.pi1 * x[...,0]
        y2 = self.pi2 * x[...,1]

        y = th.bitwise_xor(y1, y2)
        y = th.remainder(y, self.table_size)
        
        return y

    def forward(self, x: th.Tensor):
        # x: (batch_size, data_dim)
        # output: (batch_size, n_features)

        batch_size, data_dim = x.shape

        # get corners:
        x_scaled = x * self.resolution
        x_floor = th.floor(x_scaled)
        # x_ceil = th.ceil(x_scaled)
        x_ceil = x_floor + 1
        x_lim = th.stack((x_floor, x_ceil), dim=1)

        idx_list = [(0,0), (0,1), (1,0), (1,1)]

        corners = th.stack(
            [
                x_lim[:,[i,j], th.arange(2)] for i,j in idx_list
            ],
        dim=1).to(th.int64)


        feature_idx = self.compute_idx(corners)
        features = self.table[feature_idx]

        # get weights:
        x_diff = x_scaled.unsqueeze(1) - corners
        x_diff = th.abs(x_diff)
        weights = 1 - x_diff
        weights = th.prod(weights, dim=-1)

        # get output:
        output = th.sum(features * weights.unsqueeze(-1), dim=1)

        return output

        

class INGPEncoding(nn.Module):
    def __init__(self, resolution_max, resolution_min,
                 table_size, n_features, n_levels,
                 pi1=1, pi2=2654435761):
        super().__init__()
        self.output_dim = n_features*n_levels
        self.resolution_max = resolution_max
        self.resolution_min = resolution_min
        self.table_size = table_size
        self.n_features = n_features
        self.n_levels = n_levels
        self.b = 1 if n_levels==1 else math.exp((math.log(resolution_max) - math.log(resolution_min)) / (n_levels-1))

        self.resolution = th.floor(resolution_min * self.b**th.arange(n_levels))

        self.encodings = nn.ModuleList(
            [INGPTable(int(r), table_size, n_features, pi1, pi2) for r in self.resolution]
        )
    
    def forward(self, x):
        # x: (batch_size, data_dim)
        # output: (batch_size, n_features*n_levels)

        output = th.cat([enc(x) for enc in self.encodings], dim=1)

        return output


class Gigapixel(pl.LightningModule):
    def __init__(self, n_hidden: int,
                 hidden_dim: int,
                 position_encoder: INGPEncoding,):
        super().__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.position_encoder = position_encoder

        if self.n_hidden == 0:
            self.net = nn.Linear(self.position_encoder.output_dim, 3)
        else:
            layer1 = nn.Linear(self.position_encoder.output_dim, hidden_dim)
            layer2 = nn.Linear(hidden_dim, 3)
            intermediate_layers = []
            for _ in range(self.n_hidden-1):
                intermediate_layers += [nn.ReLU(True), nn.Linear(hidden_dim, hidden_dim)]
            self.net = nn.Sequential(layer1, *intermediate_layers, nn.ReLU(True), layer2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, pos: th.Tensor):
        pos = self.position_encoder(pos)
        
        z = self.net(pos)
        rgb = self.sigmoid(z)

        return rgb


    def step_helper(self, stage: Literal["train", "val", "test"], batch: th.Tensor, batch_idx: int):
        pos, rgb = batch
        rgb_pred = self(pos)
        loss = nn.functional.mse_loss(rgb_pred, rgb)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch: th.Tensor, batch_idx: int):
        return self.step_helper("train", batch, batch_idx)
    
    def validation_step(self, batch: th.Tensor, batch_idx: int):
        return self.step_helper("val", batch, batch_idx)
    
    def test_step(self, batch: th.Tensor, batch_idx: int):
        return self.step_helper("test", batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"
        }
    
