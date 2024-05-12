import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(True),
        )

        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(2, 2)
        # print("Down out_channels is",out_channels)

    def forward(self, x):
        z = self.conv(x)
        y = self.alpha * self.maxpool(z) + (1 - self.alpha) * self.avgpool(z)
        return y, z


class Bottom(nn.Module):
    def __init__(self, in_channels, qk_dim, v_dim, out_channels):
        super().__init__()

        self.qkv_conv = nn.Conv2d(in_channels, qk_dim*2 + v_dim, 2, 2)
        self.split_dims = [qk_dim, qk_dim, v_dim]
        self.scale = 1. / (qk_dim ** 0.5)
        
        self.norm = nn.LayerNorm(v_dim, eps=1e-6)
        self.o_conv = nn.ConvTranspose2d(v_dim, out_channels, 2, 2)
        self.act = nn.GELU()

    def forward(self, x):
        B, H, W = x.shape[0], *x.shape[2:]
        q, k, v = self.qkv_conv(x).permute(0, 2, 3, 1).flatten(1, 2).split(self.split_dims, dim=-1)
        y = (q @ k.transpose(1, 2) * self.scale).softmax(dim=-1) @ v
        y = self.norm(y).view(B, H*W//4, -1).transpose(1, 2).reshape(B, -1, H//2, W//2)

        o = self.act(self.o_conv(y))
        return o


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels, eps=1e-6),
            nn.ReLU(True),
        )

    def forward(self, x, z):
        x = self.up(x)
        y = torch.cat([x, z], dim=1)
        return self.conv(y)


class Unet(nn.Module):
    def __init__(self, in_channels, start_channels, out_channels, n_steps):
        super().__init__()

        self.Downs = nn.ModuleList()
        self.Ups = nn.ModuleList()
        self.Z = []
        mid_channels = start_channels

        self.Downs.append(Down(in_channels, mid_channels))
        for _ in range(n_steps - 1):
            self.Downs.append(Down(mid_channels, mid_channels*2))
            mid_channels *= 2
        
        self.bottom = Bottom(mid_channels, mid_channels, mid_channels*4, mid_channels*2)
        mid_channels *= 2
        
        for _ in range(n_steps):
            self.Ups.append(Up(mid_channels, mid_channels//2))
            mid_channels //= 2

        self.cls_head = nn.Conv2d(mid_channels, out_channels, 1)
    
    def forward(self, x):
        for down in self.Downs:
            x, z = down(x)
            self.Z.append(z)
        
        x = self.bottom(x)

        for up in self.Ups:
            x = up(x, self.Z[-1])
            self.Z.pop()
        
        return self.cls_head(x)

