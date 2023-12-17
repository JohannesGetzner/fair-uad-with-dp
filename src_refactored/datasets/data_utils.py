from typing import Any, List, Tuple, Callable
from torch.utils.data import default_collate
import pandas as pd
import os
import numpy as np
import json
from torch import Tensor
import pydicom as dicom
import torch
from torchvision import transforms
from tqdm import tqdm


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
    return img[None]


def default_load_func(x):
    return torch.tensor(x)


def get_load_fn(dataset: str):
    if dataset == 'rsna':
        return load_dicom_img
    elif dataset in ['cxr14', 'mimic-cxr', 'chexpert']:
        return default_load_func
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


def get_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=False),
        transforms.Normalize([0.5], [0.5])
    ])


def read_memmap(filename: str) -> np.ndarray:
    """
    Read NumPy memmap file with separate JSON metadata file.

    Args:
        filename (str): File name for the memmap file (without extension).

    Returns:
        numpy.ndarray: Loaded NumPy array.
    """
    # Read metadata JSON file
    metadata_file = f"{filename}.json"
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])

    # Read memmap file
    memmap_file = f"{filename}.dat"
    data = np.memmap(memmap_file, dtype=dtype, shape=shape, mode='r')

    return data


def write_memmap(files: List[str], filename: str, load_fn: Callable, target_size: Tuple[int, int]):
    """Write NumPy memmap file with separate JSON metadata file."""
    memmap_file = f"{filename}.dat"
    shape = (len(files), 1, *target_size)
    dtype = 'float32'
    fp = np.memmap(memmap_file, dtype=dtype, mode='w+', shape=shape)

    n_failed = 0
    failed = []
    for i, file in tqdm(enumerate(files), total=len(files)):
        try:
            fp[i] = load_fn(file)
        except Exception as e:
            n_failed += 1
            failed.append(file)
            print(f"Failed to load image '{file}': {e}")
            fp[i] = np.zeros(shape[1:], dtype=dtype)
            continue
    fp.flush()

    # Write metadata JSON file
    metadata = {
        'shape': shape,
        'dtype': str(dtype)
    }
    metadata_file = f"{filename}.json"
    with open(metadata_file, 'w') as file:
        json.dump(metadata, file)

    print(f"Failed to load {n_failed} images.")
    for file in failed:
        print(file)