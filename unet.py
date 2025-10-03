import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pathlib import Path
from tqdm import tqdm
import math

# --- U-Net Model for the Vector Field ---
# A U-Net is a standard choice for image generation tasks as it can
# capture multi-scale features effectively.

class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes time 't' into a high-dimensional vector."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    """A basic convolutional block with GroupNorm."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_ch),
            nn.ReLU()
        )

    def forward(self, x, t):
        h = self.norm(self.relu(self.conv1(x)))
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1) # Add time embedding
        h = self.norm(self.relu(self.conv2(h)))
        return h

class UNet(nn.Module):
    """The neural network approximating the vector field v(x_t, t)."""
    def __init__(self, img_channels=3, time_emb_dim=32):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Downsampling path
        self.down1 = Block(img_channels, 64, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(64, 128, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(128, 256, time_emb_dim)

        # Upsampling path
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = Block(256, 128, time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2 = Block(128, 64, time_emb_dim)

        # Final convolution
        self.out_conv = nn.Conv2d(64, img_channels, kernel_size=1)

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)

        # Downsampling
        x1 = self.down1(x, t_emb)
        p1 = self.pool1(x1)
        x2 = self.down2(p1, t_emb)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bot1(p2, t_emb)

        # Upsampling
        u1 = self.upconv1(b)
        # Skip connection from corresponding downsampling layer
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1(u1, t_emb)

        u2 = self.upconv2(u1)
        # Skip connection
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2(u2, t_emb)

        # Final output
        return self.out_conv(u2)