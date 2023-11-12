import os
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import pandas as pd
from torch import Tensor, Generator
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms
import json
import sys
sys.path.append("..")
from src import RSNA_DIR
import numpy as np
from src.data.data_utils import load_dicom_img
from src.data.rsna_pneumonia_detection import (load_rsna_age_two_split,
                                               load_rsna_gender_split,
                                               load_rsna_naive_split,
                                               load_rsna_intersectional_age_sex_split
                                               )
from src import CHEXPERT_DIR, CXR14_DIR, MIMIC_CXR_DIR
from src.data.chexpert import (load_chexpert_age_split,
                               load_chexpert_naive_split,
                               load_chexpert_race_split,
                               load_chexpert_sex_split,
                               load_chexpert_intersectional_age_sex_split
                               )
from src.data.cxr14 import (load_cxr14_age_split,
                            load_cxr14_naive_split,
                            load_cxr14_sex_split,
                            load_cxr14_intersectional_age_sex_split
                            )
from src.data.mimic_cxr import (load_mimic_cxr_age_split,
                                load_mimic_cxr_intersectional_age_sex_split,
                                load_mimic_cxr_naive_split,
                                load_mimic_cxr_race_split,
                                load_mimic_cxr_sex_split)


class NormalDataset_rsna(Dataset):
    """
    Anomaly detection training dataset.
    Receives a list of filenames
    """

    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            load_fn: Callable = load_dicom_img):
        """
        :param filenames: Paths to training images
        :param gender:
        """
        self.data = data
        self.filenames = data.copy()
        for i, d in enumerate(self.data):
            if isinstance(d, str):
                self.data[i] = transform(load_fn(d))
        self.labels = labels
        self.meta = meta
        self.load_fn = load_fn
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        img = self.data[idx]
        label = self.labels[idx]
        meta = self.meta[idx]
        return img, label, meta


class AnomalFairnessDataset_rsna(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(
            self,
            data: Dict[str, List[str]],
            labels: Dict[str, List[int]],
            meta: Dict[str, List[int]],
            transform=None,
            load_fn: Callable = load_dicom_img):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn

    def __len__(self) -> int:
        return min([len(v) for v in self.data.values()])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(self.load_fn(v[idx])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta


class NormalDataset_other(Dataset):
    """
    Anomaly detection training dataset.
    Receives a list of filenames
    """

    def __init__(
            self,
            data: List[str],
            labels: List[int],
            meta: List[int],
            transform=None,
            index_mapping: Optional[List[int]] = None,
            load_fn: Callable = lambda x: x,
            filenames = None
    ):
        """
        :param filenames: Paths to training images
        :param labels: Class labels (0 == normal, other == anomaly)
        :param meta: Metadata (such as age or sex labels)
        :param transform: Transformations to apply to images
        :param index_mapping: Mapping from indices to data
        """
        self.data = data
        self.labels = labels
        self.meta = meta
        self.load_fn = load_fn
        self.transform = transform
        self.index_mapping = index_mapping
        self.filenames = filenames

        if self.index_mapping is None or (len(index_mapping) == len(self.data)):
            self.index_mapping_cpy = index_mapping
            self.index_mapping = torch.arange(len(self.data))

        for i, d in enumerate(self.data):
            if isinstance(d, str):
                self.data[i] = transform(load_fn(d))
                self.load_fn = lambda x: x
                self.transform = lambda x: x

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tensor:
        data_idx = self.index_mapping[idx]
        img = self.transform(self.load_fn(self.data[data_idx]))
        label = self.labels[idx]
        meta = self.meta[idx]
        return img, label, meta


class AnomalFairnessDataset_other(Dataset):
    """
    Anomaly detection test dataset.
    Receives a list of filenames and a list of class labels (0 == normal).
    """
    def __init__(
            self,
            data: Dict[str, List[str]],
            labels: Dict[str, List[int]],
            meta: Dict[str, List[int]],
            transform=None,
            index_mapping: Optional[Dict[str, List[int]]] = None,
            load_fn: Callable = lambda x: x):
        """
        :param filenames: Paths to images for each subgroup
        :param labels: Class labels for each subgroup (0 == normal, other == anomaly)
        """
        super().__init__()
        self.data = data
        self.labels = labels
        self.meta = meta
        self.transform = transform
        self.load_fn = load_fn
        self.index_mapping = index_mapping

        if self.index_mapping is None:
            self.index_mapping = {mode: torch.arange(len(self.data)) for mode in self.data.keys()}

    def __len__(self) -> int:
        return min([len(v) for v in self.labels.values()])

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = {k: self.transform(self.load_fn(v[self.index_mapping[k][idx]])) for k, v in self.data.items()}
        label = {k: v[idx] for k, v in self.labels.items()}
        meta = {k: v[idx] for k, v in self.meta.items()}
        return img, label, meta


# default_collate does not work with Lists of dictionaries
def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 3
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}
    meta = {k: default_collate([sample[2][k] for sample in batch]) for k in keys}

    return imgs, labels, meta

def get_dataloaders_other(dataset: str,
                          batch_size: int,
                          img_size: int,
                          protected_attr: str,
                          num_workers: Optional[int] = 4,
                          male_percent: Optional[float] = 0.5,
                          old_percent: Optional[float] = 0.5,
                          white_percent: Optional[float] = 0.5,
                          n_training_samples: int = None,
                          max_train_samples: Optional[int] = None,
                          train_dataset_mode = "") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for the desired dataset.
    """
    print(f'Loading dataset {dataset} with protected attribute {protected_attr}')

    # Load filenames and labels
    filenames = None
    if dataset == 'cxr14':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_cxr14_naive_split()
            filenames = None
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_cxr14_sex_split(
                cxr14_dir=CXR14_DIR,
                male_percent=male_percent)
        elif protected_attr == 'age':
            data, labels, meta, idx_map, filenames = load_cxr14_age_split(
                cxr14_dir=CXR14_DIR,
                old_percent=old_percent)
        elif protected_attr == 'balanced':
            data, labels, meta, idx_map, filenames = load_cxr14_intersectional_age_sex_split(
                cxr14_dir = CXR14_DIR
            )
        else:
            raise NotImplementedError
    elif dataset == 'mimic-cxr':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_mimic_cxr_naive_split(
                max_train_samples=max_train_samples)
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_mimic_cxr_sex_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                male_percent=male_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'age':
            data, labels, meta, idx_map = load_mimic_cxr_age_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                old_percent=old_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'race':
            data, labels, meta, idx_map = load_mimic_cxr_race_split(
                mimic_cxr_dir=MIMIC_CXR_DIR,
                white_percent=white_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'balanced':
            data, labels, meta, idx_map = load_mimic_cxr_intersectional_age_sex_split(
                mimic_cxr_dir=MIMIC_CXR_DIR)
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    elif dataset == 'chexpert':
        def load_fn(x):
            return torch.tensor(x)
        if protected_attr == 'none':
            data, labels, meta, idx_map = load_chexpert_naive_split(
                chexpert_dir=CHEXPERT_DIR,
                max_train_samples=max_train_samples)
        elif protected_attr == 'sex':
            data, labels, meta, idx_map = load_chexpert_sex_split(
                chexpert_dir=CHEXPERT_DIR,
                male_percent=male_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'age':
            data, labels, meta, idx_map = load_chexpert_age_split(
                chexpert_dir=CHEXPERT_DIR,
                old_percent=old_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'race':
            data, labels, meta, idx_map = load_chexpert_race_split(
                chexpert_dir=CHEXPERT_DIR,
                white_percent=white_percent,
                max_train_samples=max_train_samples)
        elif protected_attr == 'balanced':
            data, labels, meta, idx_map, filenames = load_chexpert_intersectional_age_sex_split(
                chexpert_dir=CHEXPERT_DIR
            )
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    elif dataset == 'rsna':
        def load_fn(x):
            return torch.tensor(x)
        data, labels, meta, idx_map, filenames = load_rsna_intersectional_age_sex_split(
            RSNA_DIR
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train_data = data['train']
    train_labels = labels['train']
    train_meta = meta['train']
    train_idx_map = idx_map['train']
    if filenames is not None:
        train_filenames = filenames['train']
    else:
        train_filenames = None
    val_data = {k: v for k, v in data.items() if 'val' in k}
    val_labels = {k: v for k, v in labels.items() if 'val' in k}
    val_meta = {k: v for k, v in meta.items() if 'val' in k}
    val_idx_map = {k: v for k, v in idx_map.items() if 'val' in k}
    test_data = {k: v for k, v in data.items() if 'test' in k}
    test_labels = {k: v for k, v in labels.items() if 'test' in k}
    test_meta = {k: v for k, v in meta.items() if 'test' in k}
    test_idx_map = {k: v for k, v in idx_map.items() if 'test' in k}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=False),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    anomal_ds = partial(AnomalFairnessDataset_other, transform=transform, load_fn=load_fn)
    val_dataset = anomal_ds(val_data, val_labels, val_meta, index_mapping=val_idx_map)
    test_dataset = anomal_ds(test_data, test_labels, test_meta, index_mapping=test_idx_map)

    train_dataloaders = []
    if n_training_samples:
        print("ATTENTION: n_training_samples is set to", n_training_samples)
        print("Splitting training set into", len(train_meta) // n_training_samples, "dataloaders")
        prev = 0
        for i in range(n_training_samples, len(train_meta), n_training_samples):
            temp_dataset = NormalDataset_other(
                train_data[train_idx_map[prev:i]],
                train_labels[prev:i],
                train_meta[prev:i],
                transform=transform,
                index_mapping=train_idx_map[prev:i],
                load_fn=load_fn,
                filenames=train_filenames[prev:i]
            )
            temp_dataloader = DataLoader(
                temp_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                generator=Generator().manual_seed(2147483647)
            )
            train_dataloaders.append(temp_dataloader)
            prev = i
    elif train_dataset_mode.startswith("best"):
        with open('subsets.json', 'r') as f:
            best_samples = json.load(f)
        if "-" in train_dataset_mode:
            scoring_metric = train_dataset_mode.split("-")[1]
            subset_sizes = [1, 5, 10, 25, 50]
        else:
            scoring_metric = "test/AUROC"
            subset_sizes = [1, 5, 10, 25, 50, 100, 250, 500]
        best_samples = best_samples[scoring_metric]
        for subset_size in subset_sizes:
            temp_df = pd.DataFrame.from_dict(best_samples)
            temp_df = temp_df.sort_values(by=['scores'], ascending=False)
            subset_idx_map = temp_df["idx_map"].iloc[:subset_size].to_list()
            subset_labels = temp_df["labels"].iloc[:subset_size].to_list()
            subset_meta = temp_df["meta"].iloc[:subset_size].to_list()
            subset_filenames = temp_df["filenames"].iloc[:subset_size].to_list()

            temp_dataset = NormalDataset_other(
                train_data[subset_idx_map],
                subset_labels,
                subset_meta,
                transform=transform,
                index_mapping=subset_idx_map,
                load_fn=load_fn,
                filenames=subset_filenames
            )
            temp_dataloader = DataLoader(
                temp_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                generator=Generator().manual_seed(2147483647)
                )
            train_dataloaders.append(temp_dataloader)
    elif train_dataset_mode == "random":
        for subset_size in [1, 5, 10, 25, 50, 100, 250, 500]:
            # get random subset of train_idx_map list
            random_indices = np.random.choice(len(train_idx_map), subset_size, replace=False)
            subset_labels = [train_labels[i] for i in random_indices]
            subset_meta = [train_meta[i] for i in random_indices]
            subset_filenames = [train_filenames[i] for i in random_indices]
            subset_idx_map = [train_idx_map[i] for i in random_indices]

            train_dataset = NormalDataset_other(
                train_data,
                subset_labels,
                subset_meta,
                transform=transform,
                index_mapping=subset_idx_map,
                load_fn=load_fn,
                filenames=subset_filenames
            )
            train_dataloaders.append(
                DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    generator=Generator().manual_seed(2147483647),
                    pin_memory=True)
            )
    else:
        train_dataset = NormalDataset_other(
            train_data,
            train_labels,
            train_meta,
            transform=transform,
            index_mapping=train_idx_map,
            load_fn=load_fn,
            filenames=train_filenames
        )
        train_dataloaders.append(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=Generator().manual_seed(2147483647),
            pin_memory=True))
    dl = partial(
        DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=group_collate_fn,
        pin_memory=True)
    val_dataloader = dl(
        val_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))
    test_dataloader = dl(
        test_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))

    return (train_dataloaders,
            val_dataloader,
            test_dataloader)

def get_dataloaders_rsna(dataset: str,
                         batch_size: int,
                         img_size: int,
                         protected_attr: str,
                         num_workers: Optional[int] = 4,
                         male_percent: Optional[float] = 0.5,
                         old_percent: Optional[float] = 0.5,
                         upsampling_strategy=None,
                         effective_dataset_size=1.0,
                         random_state=42,
                         n_training_samples: int = None,
                         ) -> Tuple[List[DataLoader], DataLoader, DataLoader, int]:
    """
    Returns dataloaders for the RSNA dataset.
    """
    # Load filenames and labels
    if dataset == 'rsna':
        load_fn = load_dicom_img
        if protected_attr == 'none':
            data, labels, meta = load_rsna_naive_split(RSNA_DIR)
        elif protected_attr == 'age':
            data, labels, meta = load_rsna_age_two_split(
                RSNA_DIR,
                old_percent=old_percent,
                upsampling_strategy=upsampling_strategy,
                effective_dataset_size=effective_dataset_size,
                random_state=random_state
            )
        elif protected_attr == 'sex':
            data, labels, meta = load_rsna_gender_split(
                RSNA_DIR,
                male_percent=male_percent,
                upsampling_strategy=upsampling_strategy,
                effective_dataset_size=effective_dataset_size,
                random_state=random_state
            )
        else:
            raise ValueError(f'Unknown protected attribute: {protected_attr} for dataset {dataset}')
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    train_data = data['train']
    train_labels = labels['train']
    train_meta = meta['train']

    if upsampling_strategy:
        # get the max number of times a sample appears in train_data
        counts = Counter(train_data)
        max_count = max(counts.values())
        print(f"max sample replication is {max_count}")
    else:
        max_count = 1

    val_data = {k: v for k, v in data.items() if 'val' in k}
    val_labels = {k: v for k, v in labels.items() if 'val' in k}
    val_meta = {k: v for k, v in meta.items() if 'val' in k}
    test_data = {k: v for k, v in data.items() if 'test' in k}
    test_labels = {k: v for k, v in labels.items() if 'test' in k}
    test_meta = {k: v for k, v in meta.items() if 'test' in k}

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=False),
        transforms.Normalize([0.5], [0.5])
    ])

    # Create datasets
    anomal_ds = partial(AnomalFairnessDataset_rsna, transform=transform, load_fn=load_fn)
    val_dataset = anomal_ds(val_data, val_labels, val_meta)
    test_dataset = anomal_ds(test_data, test_labels, test_meta)

    train_dataset = NormalDataset_rsna(train_data, train_labels, train_meta, transform=transform, load_fn=load_fn)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=Generator().manual_seed(2147483647))
    dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=group_collate_fn)
    val_dataloader = dl(
        val_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))
    test_dataloader = dl(
        test_dataset,
        shuffle=False,
        generator=Generator().manual_seed(2147483647))

    return (train_dataloader,
            val_dataloader,
            test_dataloader, max_count)
