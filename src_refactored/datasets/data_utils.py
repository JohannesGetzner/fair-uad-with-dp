from typing import Any, List, Tuple
from torch.utils.data import default_collate
import pandas as pd
import os
from torch import Tensor
import pydicom as dicom
import torch
from torchvision import transforms

def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 3
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}
    meta = {k: default_collate([sample[2][k] for sample in batch]) for k in keys}

    return imgs, labels, meta

def load_dicom_img(filename: str) -> Tensor:
    """Loads a DICOM image."""
    ds = dicom.dcmread(filename)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32) / 255.0
    return img[None]  #

def load_rsna_files(anomaly = 'lungOpacity'):
    metadata = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csvs', 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        anomalous_data = metadata[metadata.label == 1]
    else:
        anomalous_data = metadata[metadata.label == 2]
    return normal_data, anomalous_data

def get_load_fn(dataset: str):
    if dataset == 'rsna':
        return load_dicom_img
    else:
        raise ValueError(f"Dataset {dataset} not supported.")

def get_transforms(dataset:str, img_size: int):
    if dataset == 'rsna':
        return transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=False),
            transforms.Normalize([0.5], [0.5])
        ])