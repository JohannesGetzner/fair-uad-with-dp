from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import Tensor, Generator
from torch.utils.data import DataLoader, Dataset, default_collate
from torchvision import transforms

from src import RSNA_DIR
from src.data.data_utils import load_dicom_img
from src.data.rsna_pneumonia_detection import (load_rsna_age_two_split,
                                               load_rsna_gender_split,
                                               load_rsna_naive_split)


class NormalDataset(Dataset):
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


class AnomalFairnessDataset(Dataset):
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


# default_collate does not work with Lists of dictionaries
def group_collate_fn(batch: List[Tuple[Any, ...]]):
    assert len(batch[0]) == 3
    keys = batch[0][0].keys()

    imgs = {k: default_collate([sample[0][k] for sample in batch]) for k in keys}
    labels = {k: default_collate([sample[1][k] for sample in batch]) for k in keys}
    meta = {k: default_collate([sample[2][k] for sample in batch]) for k in keys}

    return imgs, labels, meta


def get_dataloaders(dataset: str,
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
                    best_and_worst_subsets: bool = False
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
    anomal_ds = partial(AnomalFairnessDataset, transform=transform, load_fn=load_fn)
    val_dataset = anomal_ds(val_data, val_labels, val_meta)
    test_dataset = anomal_ds(test_data, test_labels, test_meta)

    train_dataloaders = []
    if n_training_samples:
        print("ATTENTION: n_training_samples is set to", n_training_samples)
        print("splitting training set into", len(train_data) // n_training_samples, "dataloaders")
        prev = 0
        for i in range(n_training_samples, len(train_data), n_training_samples):
            temp_dataset = NormalDataset(
                train_data[prev:i],
                train_labels[prev:i],
                train_meta[prev:i],
                transform=transform,
                load_fn=load_fn
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
    elif best_and_worst_subsets:
        # load subsets.json
        import json
        with open('logs_persist/distillation/subsets.json', 'r') as f:
            subsets = json.load(f)
            for subset in subsets:
                # get indices of filenames in subset from train_data
                indices = [train_data.index(filename) for filename in subset["filenames"]]
                temp_dataset = NormalDataset(
                    [train_data[i] for i in indices],
                    [train_labels[i] for i in indices],
                    [train_meta[i] for i in indices],
                    transform=transform,
                    load_fn=load_fn
                )
                temp_dataloader = DataLoader(
                    temp_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    generator=Generator().manual_seed(2147483647)
                )
                train_dataloaders.append(temp_dataloader)
    else:
        train_dataset = NormalDataset(train_data, train_labels, train_meta, transform=transform, load_fn=load_fn)
        train_dataloaders.append(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=Generator().manual_seed(2147483647)))
    dl = partial(DataLoader, batch_size=batch_size, num_workers=num_workers, collate_fn=group_collate_fn)
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
            test_dataloader, max_count)
