from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ks: int,
            pool: bool = True,
            bias: bool = False) -> None:
        super().__init__()
        pad = ks // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ks, stride=1,
                              padding=pad, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.act = nn.ReLU(inplace=True)
        if pool:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if hasattr(self, 'pool'):
            x = self.pool(x)
        return x


class DeepSVDD(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.repr_dim = config.repr_dim

        # Input size: 1 x 128 x 128
        self.conv_in = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=False),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            ConvBlock(1, 32, 3, bias=False),  # (32, 64, 64)
            ConvBlock(32, 64, 3, bias=False),  # (64, 32, 32)
            ConvBlock(64, 128, 3, bias=False),  # (128, 16, 16)
            ConvBlock(128, 256, 3, bias=False),  # (256, 8, 8)
            ConvBlock(256, 512, 3, pool=False, bias=False),  # (512, 8, 8)
        ])
        self.linear_out = nn.Linear(512, self.repr_dim, bias=False)

        # hypersphere center c
        self.register_buffer('c', torch.randn(1, self.repr_dim) + 10)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=(2, 3))  # Global average pooling
        x = self.linear_out(x)  # (N, repr_dim)
        return x

    def loss(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Compute DeepSVDD loss."""
        pred = self.forward(x)
        loss = one_class_scores(pred, self.c).mean()
        return {
            'loss': loss,
            'oc_loss': loss,
        }

    def predict_anomaly(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute DeepSVDD loss."""
        with torch.no_grad():
            pred = self.forward(x)
        anomaly_scores = one_class_scores(pred, self.c)
        return None, anomaly_scores  # No anomaly map for DeepSVDD

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))


def one_class_scores(pred: Tensor, c: Tensor) -> Tensor:
    """Compute anomaly_score for the one-class objective."""
    return torch.sum((pred - c) ** 2, dim=1)  # (N,)


if __name__ == '__main__':
    from argparse import Namespace
    config = Namespace(
        repr_dim=128
    )
    model = DeepSVDD(config)
    model.predict_anomaly(torch.randn(2, 1, 128, 128))
