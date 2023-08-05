from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models as tv_models

from src.models.pytorch_ssim import SSIMLoss

""""""""""""""""""""""" Feature Extractor """""""""""""""""""""""""""


RESNETLAYERS = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def _set_requires_grad_false(layer):
    for param in layer.parameters():
        param.requires_grad = False


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, resnet, layer_names: List[str] = RESNETLAYERS):
        """
        Returns features on multiple levels from a ResNet18.
        Available layers: 'layer0', 'layer1', 'layer2', 'layer3', 'layer4',
                          'avgpool'
        Args:
            resnet (nn.Module): Type of resnet used
            layer_names (list): List of string of layer names where to return
                                the features. Must be ordered
            pretrained (bool): Whether to load pretrained weights
        Returns:
            out (dict): Dictionary containing the extracted features as
                        torch.tensors
        """
        super().__init__()

        _set_requires_grad_false(resnet)

        # [b, 3, 256, 256]
        self.layer0 = nn.Sequential(
            *list(resnet.children())[:4])  # [b, 64, 64, 64]
        self.layer1 = resnet.layer1  # [b, 64, 64, 64]
        self.layer2 = resnet.layer2  # [b, 128, 32, 32]
        self.layer3 = resnet.layer3  # [b, 256, 16, 16]
        self.layer4 = resnet.layer4  # [b, 512, 8, 8]
        self.avgpool = resnet.avgpool  # [b, 512, 1, 1]

        self.layer_names = layer_names

    def forward(self, inp: Tensor) -> Dict[str, Tensor]:
        if inp.shape[1] == 1:
            inp = inp.repeat(1, 3, 1, 1)

        out = {}
        for name, module in self._modules.items():
            inp = module(inp)
            if name in self.layer_names:
                out[name] = inp
            if name == self.layer_names[-1]:
                break
        return out


class ResNet18FeatureExtractor(ResNetFeatureExtractor):
    def __init__(self, layer_names: List[str] = RESNETLAYERS,
                 pretrained: bool = True):
        weights = 'IMAGENET1K_V1' if pretrained else None
        super().__init__(tv_models.resnet18(weights=weights), layer_names)


class Extractor(nn.Module):
    """
    Muti-scale regional feature based on VGG-feature maps.
    """

    def __init__(
        self,
        cnn_layers: List[str] = ['layer1', 'layer2'],
        upsample: str = 'bilinear',
        inp_size: int = 128,
        keep_feature_prop: float = 1.0,
        pretrained: bool = True,
    ):
        super().__init__()

        self.backbone = ResNet18FeatureExtractor(layer_names=cnn_layers,
                                                 pretrained=pretrained)
        self.inp_size = inp_size
        self.featmap_size = inp_size // 4
        self.upsample = upsample
        self.align_corners = True if upsample == "bilinear" else None

        # Find out how many channels we got from the backbone
        c_feats = self.get_out_channels()

        # Create mask to drop random features_channels
        self.register_buffer('feature_mask', torch.Tensor(
            c_feats).uniform_() < keep_feature_prop)
        self.c_feats = self.feature_mask.sum().item()

    def get_out_channels(self) -> int:
        """Get the number of channels of the output feature map"""
        device = next(self.backbone.parameters()).device
        x = torch.randn((2, 1, self.inp_size, self.inp_size), device=device)
        return sum([feat_map.shape[1] for feat_map in self.backbone(x).values()])

    def forward(self, inp: Tensor) -> Tensor:
        # Center input
        inp = (inp - 0.5) * 2

        # Extract feature maps
        feat_maps = self.backbone(inp)

        features = []
        for feat_map in feat_maps.values():
            # Resizing
            feat_map = F.interpolate(feat_map, size=self.featmap_size,
                                     mode=self.upsample,
                                     align_corners=self.align_corners)
            features.append(feat_map)

        # Concatenate to tensor
        features = torch.cat(features, dim=1)

        # Drop out feature maps
        features = features[:, self.feature_mask]

        return features


""""""""""""""""""""""" Feature Autoencoder """""""""""""""""""""""""""


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ks: int,
            norm_layer: Optional[str] = None,
            dropout: Optional[float] = 0.0,
            bias: bool = False) -> None:
        super().__init__()
        pad = ks // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ks, stride=2,
                              padding=pad, bias=bias)
        if norm_layer is not None:
            self.norm = eval(norm_layer)(out_channels)
        self.act = nn.LeakyReLU()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = self.act(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ks: int,
            pad: int,
            norm_layer: Optional[str] = None,
            dropout: Optional[float] = 0.0,
            bias: bool = False) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, ks, stride=2,
                                       padding=pad, bias=bias)
        if norm_layer is not None:
            self.norm = eval(norm_layer)(out_channels)
        self.act = nn.LeakyReLU()
        if dropout > 0.0:
            self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = self.act(x)
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x


class VanillaFeatureEncoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_dims: List[int],
            norm_layer: Optional[str] = None,
            dropout: Optional[float] = 0.0,
            bias: bool = False) -> None:
        super().__init__()
        ks = 5
        pad = ks // 2

        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.layers.append(EncoderBlock(in_channels, hidden_dims[i], ks,
                                            norm_layer, dropout, bias))
            in_channels = hidden_dims[i]
        self.out = nn.Conv2d(in_channels, in_channels, ks, stride=1,
                             padding=pad, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


class VanillaFeatureDecoder(nn.Module):
    def __init__(
            self,
            out_channels: int,
            hidden_dims: List[int],
            norm_layer: Optional[str] = None,
            dropout: Optional[float] = 0.0,
            bias: bool = False) -> None:
        super().__init__()
        ks = 6
        pad = 2

        hidden_dims = [out_channels] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.layers.append(DecoderBlock(hidden_dims[i], hidden_dims[i - 1],
                                            ks, pad, norm_layer, dropout, bias))
        self.out = nn.Conv2d(hidden_dims[0], out_channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


class FeatureReconstructor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.extractor = Extractor(inp_size=config.img_size,
                                   cnn_layers=config.extractor_cnn_layers,
                                   keep_feature_prop=config.keep_feature_prop,
                                   pretrained=True)

        config.in_channels = self.extractor.c_feats
        self.enc = VanillaFeatureEncoder(config.in_channels,
                                         config.hidden_dims,
                                         norm_layer="nn.BatchNorm2d",
                                         dropout=config.dropout,
                                         bias=False)
        self.dec = VanillaFeatureDecoder(config.in_channels,
                                         config.hidden_dims,
                                         norm_layer="nn.BatchNorm2d",
                                         dropout=config.dropout,
                                         bias=False)

        if config.loss_fn == 'ssim':
            self.loss_fn = SSIMLoss(window_size=5, size_average=False)
        elif config.loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss function: {config.loss_fn}")
        self.weigh_loss = config.weigh_loss

    def forward(self, x: Tensor):
        feats = self.get_feats(x)
        rec = self.get_rec(feats)
        return feats, rec

    def get_feats(self, x: Tensor, **kwargs) -> Tensor:
        with torch.no_grad():
            feats = self.extractor(x)
        return feats

    def get_rec(self, feats: Tensor) -> Tensor:
        z = self.enc(feats)
        rec = self.dec(z)
        return rec

    def loss(self, x: Tensor, **kwargs):
        if self.weigh_loss is not None and "per_sample_loss_weights" in kwargs.keys():
            return self.weighted_loss(x, **kwargs)
        feats, rec = self(x)
        loss = self.loss_fn(rec, feats).mean()
        return {'loss': loss, 'rec_loss': loss}

    def weighted_loss(self, x: Tensor, **kwargs):
        feats, rec = self(x)
        loss = self.loss_fn(rec, feats)
        expanded_loss_weights = kwargs["per_sample_loss_weights"].view(-1, 1, 1, 1)
        weighted_loss = (loss * expanded_loss_weights)
        min_loss = torch.min(weighted_loss)
        max_loss = torch.max(weighted_loss)
        scaled_weighted_loss = 2 * (weighted_loss - min_loss) / (max_loss - min_loss)
        mean_weighted_loss = scaled_weighted_loss.mean()
        return {'loss': mean_weighted_loss, 'rec_loss': mean_weighted_loss}

    def predict_anomaly(self, x: Tensor, **kwargs):
        """Returns per image anomaly maps and anomaly scores"""
        # Extract features
        feats, rec = self(x)

        # Compute anomaly map
        anomaly_map = self.loss_fn(rec, feats).mean(1, keepdim=True)
        # up-sample to same size as x
        anomaly_map = F.interpolate(anomaly_map, x.shape[-2:], mode='bilinear',
                                    align_corners=True)

        # Anomaly score only where object in the image, i.e. at x > 0
        anomaly_score = []
        # iterate over all samples in batch
        for i in range(x.shape[0]):
            # roi are those regions where x has "content"
            roi = anomaly_map[i][x[i] > 0]
            # only take the top 10% of the values to compute the mean
            roi = roi[roi > torch.quantile(roi, 0.9)]
            anomaly_score.append(roi.mean())
        anomaly_score = torch.stack(anomaly_score)
        # the training loss, the anomaly map and the anomaly score are all derived from the loss function
        # if the loss is high -> so is the anomaly map and the anomaly score
        return anomaly_map, anomaly_score

    def save(self, path: str):
        """Save the model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load weights"""
        self.load_state_dict(torch.load(path))
