import os
import numpy as np
import pandas as pd
from torch import Generator
from functools import partial

from torch.utils.data import DataLoader
from src_refactored.datasets.datasets import NormalDataset, AnomalFairnessDataset
from src_refactored.datasets.data_utils import load_rsna_files, group_collate_fn, get_load_fn, get_transforms


ATTRIBUTE_MAPPINGS = {
    "age": {"A": "old", "B": "young"},
    "sex": {"A": "male", "B": "female"}
}

DATA_DIRS = {
    "rsna": "data/RSNA"
}


class DataManager:
    def __init__(self, dataset_config):
        self.config = dataset_config

    def load_data(self):
        if self.config["dataset"] == "rsna":
            return load_rsna_files()
        else:
            raise NotImplementedError

    def establish_split(self, normal_data, anomalous_data):
        if self.config["protected_attr"] == "age":
            # Filter ages over 110 years (outliers)
            normal_data = normal_data[normal_data.PatientAge < 110]
            anomalous_data = anomalous_data[anomalous_data.PatientAge < 110]

            # Split data into bins by age
            n_bins = 3
            t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]
            self.split_info = t
            print(f"Splitting data into {n_bins - 1} bins by age. Below {np.round(t[1],2)} is young, above {np.round(t[2],2)} is old.")

            normal_young = normal_data[normal_data.PatientAge < t[1]]
            normal_old = normal_data[normal_data.PatientAge >= t[2]]
            anomalous_young = anomalous_data[anomalous_data.PatientAge < t[1]]
            anomalous_old = anomalous_data[anomalous_data.PatientAge >= t[2]]
            return normal_old, normal_young, anomalous_old, anomalous_young
        elif self.config["protected_attr"] == "sex":
            normal_male = normal_data[normal_data.PatientSex == 'M']
            normal_female = normal_data[normal_data.PatientSex == 'F']
            anomalous_male = anomalous_data[anomalous_data.PatientSex == 'M']
            anomalous_female = anomalous_data[anomalous_data.PatientSex == 'F']
            return normal_male, normal_female, anomalous_male, anomalous_female

    def encode_metadata(self, data):
        if self.config["protected_attr"] == "age":
            return np.where(data['PatientAge'] < self.split_info[1], 1, np.where(data['PatientAge'] >= self.split_info[2], 0, None))
        elif self.config["protected_attr"] == "sex":
            return np.array([0 if v == "M" else 1 for v in data['PatientSex'].values])

    def to_train_val_and_test(self, normal_A, normal_B, anomalous_A, anomalous_B):
        random_state = self.config["random_state"]
        # Save 100 young and 100 old samples for every label for validation and test
        val_test_normal_A = normal_A.sample(n=50, random_state=random_state)
        val_test_normal_B = normal_B.sample(n=50, random_state=random_state)
        val_test_anomalous_A = anomalous_A.sample(n=100,  random_state=random_state)
        val_test_anomalous_B = anomalous_B.sample(n=100,  random_state=random_state)
        val_A = val_test_anomalous_A.iloc[:50, :]
        val_B = val_test_anomalous_B.iloc[:50, :]
        test_A = val_test_anomalous_A.iloc[50:, :]
        test_B = val_test_anomalous_B.iloc[50:, :]
        # Aggregate validation and test sets and shuffle
        val_A = pd.concat([val_test_normal_A, val_A]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        val_B = pd.concat([val_test_normal_B, val_B]).sample(frac=1, random_state=random_state).reset_index(drop=True)
        test_A = pd.concat([val_test_normal_A, test_A]).sample(frac=1, random_state=random_state).reset_index( drop=True)
        test_B = pd.concat([val_test_normal_B, test_B]).sample(frac=1, random_state=random_state).reset_index(drop=True)

        rest_normal_A = normal_A[~normal_A.patientId.isin(val_test_normal_A.patientId)]
        rest_normal_B = normal_B[~normal_B.patientId.isin(val_test_normal_B.patientId)]
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

    def construct_dataloaders(self, data, labels, meta):
        train_data = data['train']
        train_labels = labels['train']
        train_meta = meta['train']

        val_data = {k: v for k, v in data.items() if 'val' in k}
        val_labels = {k: v for k, v in labels.items() if 'val' in k}
        val_meta = {k: v for k, v in meta.items() if 'val' in k}
        test_data = {k: v for k, v in data.items() if 'test' in k}
        test_labels = {k: v for k, v in labels.items() if 'test' in k}
        test_meta = {k: v for k, v in meta.items() if 'test' in k}

        # Define transforms
        transform = get_transforms(self.config["dataset"], self.config["img_size"])

        # Create datasets
        load_fn = get_load_fn(self.config["dataset"])
        anomal_ds = partial(AnomalFairnessDataset, transform=transform, load_fn=load_fn)
        val_dataset = anomal_ds(val_data, val_labels, val_meta)
        test_dataset = anomal_ds(test_data, test_labels, test_meta)

        train_dataset = NormalDataset(train_data, train_labels, train_meta, transform=transform, load_fn=load_fn)
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

    def get_dataloaders(self, custom_data_loading_hook):
        normal_data, anomalous_data = self.load_data()
        normal_A, normal_B, anomalous_A, anomalous_B = self.establish_split(normal_data, anomalous_data)
        train_A, train_B, val_A, val_B, test_A, test_B = self.to_train_val_and_test(normal_A, normal_B, anomalous_A, anomalous_B)
        train = pd.concat([*custom_data_loading_hook(train_A, train_B)]).sample(frac=1, random_state=self.config["random_state"]).reset_index(drop=True)
        filenames = {}
        labels = {}
        meta = {}
        sets = {
            f'train': train,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': val_A,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': val_B,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': test_A,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': test_B
        }
        img_dir = os.path.join(DATA_DIRS[self.config["dataset"]], 'stage_2_train_images')
        for mode, data in sets.items():
            filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
            labels[mode] = [min(1, label) for label in data.label.values]
            meta[mode] = self.encode_metadata(data)
        return self.construct_dataloaders(filenames, labels, meta)

