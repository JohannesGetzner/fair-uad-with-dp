import numpy as np
import pandas as pd
from functools import partial
from .data_utils import get_transforms, get_load_fn, group_collate_fn
from abc import ABC, abstractmethod
from .datasets import AnomalFairnessDataset, NormalDataset, MEMMAP_AnomalFairnessDataset, MEMMAP_NormalDataset
from torch.utils.data import DataLoader
from torch import Generator

ATTRIBUTE_MAPPINGS = {
    "age": {"A": "old", "B": "young"},
    "sex": {"A": "male", "B": "female"}
}

class AnomalyDataset(ABC):
    def __init__(self, dataset_config):
        self.config = dataset_config

    @abstractmethod
    def prepare_dataset(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def split_by_protected_attr(self, normal_data, anomalous_data):
        pass

    @abstractmethod
    def encode_metadata(self, data):
        pass

    @abstractmethod
    def get_dataloaders(self, custom_data_loading_hook):
        pass

    def to_train_val_and_test(self, normal_A, normal_B, anomalous_A, anomalous_B, num_normal, num_anomalous):
        random_state = self.config["random_state"]

        val_test_normal_A = normal_A.sample(n=num_normal, random_state=random_state)
        val_test_normal_B = normal_B.sample(n=num_normal, random_state=random_state)
        val_test_anomalous_A = anomalous_A.sample(n=num_anomalous,  random_state=random_state)
        val_test_anomalous_B = anomalous_B.sample(n=num_anomalous,  random_state=random_state)
        val_A = val_test_anomalous_A.iloc[:num_normal, :]
        val_B = val_test_anomalous_B.iloc[:num_normal, :]
        test_A = val_test_anomalous_A.iloc[num_normal:, :]
        test_B = val_test_anomalous_B.iloc[num_normal:, :]

        val_A = pd.concat([val_test_normal_A, val_A]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_B = pd.concat([val_test_normal_B, val_B]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_A = pd.concat([val_test_normal_A, test_A]).sample(frac=1, random_state=random_state).reset_index( drop=True)
        test_B = pd.concat([val_test_normal_B, test_B]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        rest_normal_A = normal_A[~normal_A.id.isin(val_test_normal_A.id)]
        rest_normal_B = normal_B[~normal_B.id.isin(val_test_normal_B.id)]
        print(f"Dataset Composition: {self.config['protected_attr_percent']*100}% {ATTRIBUTE_MAPPINGS[self.config['protected_attr']]['A']}")

        n_samples = min(len(rest_normal_A), len(rest_normal_B))
        n_A = int(n_samples * self.config["protected_attr_percent"])
        n_B = int(n_samples * (1 - self.config["protected_attr_percent"]))
        train_A = rest_normal_A.sample(n=n_A, random_state=random_state)
        train_B = rest_normal_B.sample(n=n_B, random_state=random_state)
        print(
            "Number of training samples for",
            f"{ATTRIBUTE_MAPPINGS[self.config['protected_attr']]['A']}/{ATTRIBUTE_MAPPINGS[self.config['protected_attr']]['B']}:",
            f"{len(train_A)}/{len(train_B)}"
        )
        print(f"Number of validation/test samples: {len(val_A)+len(val_B)}/{len(test_A)+len(test_B)}")
        return train_A, train_B, val_A, val_B, test_A, test_B

    def construct_dataloaders(self, data, labels, meta, filenames=None, idx_map=None):
        train_data = data['train']
        train_labels = labels['train']
        train_meta = meta['train']
        if filenames:
            train_filenames = filenames['train']
        val_data = {k: v for k, v in data.items() if 'val' in k}
        val_labels = {k: v for k, v in labels.items() if 'val' in k}
        val_meta = {k: v for k, v in meta.items() if 'val' in k}
        test_data = {k: v for k, v in data.items() if 'test' in k}
        test_labels = {k: v for k, v in labels.items() if 'test' in k}
        test_meta = {k: v for k, v in meta.items() if 'test' in k}
        if idx_map:
            train_idx_map = idx_map['train']
            test_idx_map = {k: v for k, v in idx_map.items() if 'test' in k}
            val_idx_map = {k: v for k, v in idx_map.items() if 'val' in k}

        # Define transforms
        transform = get_transforms(self.config["img_size"])
        # Create datasets
        load_fn = get_load_fn(self.config["dataset"])
        anomaly_dataset = partial(MEMMAP_AnomalFairnessDataset, transform=transform, load_fn=load_fn)
        val_dataset = anomaly_dataset(val_data, val_labels, val_meta, index_mapping=val_idx_map)
        test_dataset = anomaly_dataset(test_data, test_labels, test_meta, index_mapping=test_idx_map)
        train_dataset = MEMMAP_NormalDataset(
            train_data,
            train_labels,
            train_meta,
            transform=transform,
            index_mapping=train_idx_map,
            load_fn=load_fn,
            filenames=train_filenames
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            generator=Generator().manual_seed(2147483647)
        )
        dl = partial(DataLoader, batch_size=self.config["batch_size"], num_workers=4, collate_fn=group_collate_fn)
        val_dataloader = dl(val_dataset, shuffle=False, generator=Generator().manual_seed(2147483647))
        test_dataloader = dl(test_dataset, shuffle=False, generator=Generator().manual_seed(2147483647))
        return train_dataloader, val_dataloader, test_dataloader

