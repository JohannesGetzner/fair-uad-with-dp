from functools import partial
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src_refactored.models.RD.de_resnet import de_resnet18
from src_refactored.models.RD.resnet import resnet18


class ReverseDistillation(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.encoder, self.bottleneck = resnet18(pretrained=True)
        self.decoder = de_resnet18(pretrained=False)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_feats(self, x: Tensor) -> Tensor:
        n_channels = x.shape[1]
        if n_channels == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.encoder(x)
        return x

    def get_rec(self, x: Tensor) -> Tensor:
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        feats = self.get_feats(x)
        rec = self.get_rec(feats)
        return feats, rec

    def loss(self, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        feats, rec = self.forward(x)
        loss = 0.
        for f, r in zip(feats, rec):
            loss += cosine_distance(f, r).mean()
        return {
            'loss': loss,
            'cos_dist': loss,
        }

    def predict_anomaly(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        img_size = x.shape[-2:]  # (H, W)
        resize = partial(F.interpolate,
                         size=img_size,
                         mode='bilinear',
                         align_corners=False)
        anomaly_map = torch.zeros_like(x)
        with torch.no_grad():
            feats, rec = self.forward(x)
        for i, (f, r) in enumerate(zip(feats, rec)):
            cos_dist = cosine_distance(f, r)[:, None]  # (N, 1, h, w)
            anomaly_map += resize(cos_dist)  # (N, 1, H, W)
        anomaly_score = anomaly_map.amax(dim=(1, 2, 3))  # (N)
        return anomaly_map, anomaly_score

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))


def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    return 1 - F.cosine_similarity(x, y)


if __name__ == '__main__':
    model = ReverseDistillation()
    x = torch.randn(2, 1, 128, 128)
    anomaly_map, anomaly_score = model.predict_anomaly(x)
