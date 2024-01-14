import os
import numpy as np
import pandas as pd
from glob import glob
import pydicom as dicom
from tqdm import tqdm
from typing import Tuple
from .data_utils import write_memmap, read_memmap
from .anomaly_dataset import AnomalyDataset, ATTRIBUTE_MAPPINGS
from . import RSNA_DIR
from functools import partial
import torch
from torchvision import transforms as T

CLASS_MAPPING = {
    'Normal': 0,  # 8851, female: 2905, male: 4946, age mean: 44.94, std: 16.39, min: 2, max: 155
    'Lung Opacity': 1,  # 6012, female: 2502, male: 3510, age mean: 45.58, std: 17.46, min: 1, max: 92
    'No Lung Opacity / Not Normal': 2  # 11821, female: 5111, male: 6710, age mean: 49.33, std: 16.49, min: 1, max: 153
}

class RsnaAnomalyDataset(AnomalyDataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.split_info = None

    def load_data(self, anomaly="lungOpacity"):
        csv_dir = os.path.join('datasets', 'csvs', 'rsna')
        normal_data = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
        anomalous_data = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

        normal_data["id"] = normal_data["patientId"]
        anomalous_data["id"] = anomalous_data["patientId"]
        return normal_data, anomalous_data

    def split_by_protected_attr(self, normal_data, anomalous_data):
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

    def get_dataloaders(self, custom_data_loading_hook):
        normal_data, anomalous_data = self.load_data()
        normal_A, normal_B, anomalous_A, anomalous_B = self.split_by_protected_attr(normal_data, anomalous_data)
        train_A, train_B, val_A, val_B, test_A, test_B = self.to_train_val_and_test(
            normal_A,
            normal_B,
            anomalous_A,
            anomalous_B,
            num_normal=50,
            num_anomalous=100
        )
        train = pd.concat([*custom_data_loading_hook(train_A, train_B)]
        ).sample(frac=1, random_state=self.config["random_state"]).reset_index(drop=True)

        memmap_file = read_memmap(os.path.join(RSNA_DIR, 'memmap', 'data'), )
        images = {}
        labels = {}
        meta = {}
        index_mapping = {}
        filenames = {}
        # TODO: this lungOpacity should not be part of the key
        sets = {
            f'train': train,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': val_A,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': val_B,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': test_A,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': test_B}
        for mode, data in sets.items():
            images[mode] = memmap_file
            filenames[mode] = data.Path.values
            labels[mode] = [min(1, label) for label in data.label.values]
            meta[mode] = self.encode_metadata(data)
            index_mapping[mode] = data.memmap_idx.values
        return self.construct_dataloaders(images, labels, meta, filenames, index_mapping)

    def prepare_dataset(self, rsna_dir: str = RSNA_DIR):
        """Extracts metadata (labels, age, gender) from each sample of the RSNA
        dataset."""
        class_info = pd.read_csv(os.path.join(rsna_dir, 'stage_2_detailed_class_info.csv'))
        class_info.drop_duplicates(subset='patientId', inplace=True)

        metadata = []
        files = glob(f"{rsna_dir}/stage_2_train_images/*.dcm")
        for file in tqdm(files):
            ds = dicom.dcmread(file)
            patient_id = ds.PatientID
            label = class_info[class_info.patientId == patient_id]['class'].values[0]
            metadata.append(
                {'Path': file, 'patientId': patient_id, 'label': CLASS_MAPPING[label], 'PatientAge': int(ds.PatientAge),
                    'PatientSex': ds.PatientSex})

        metadata = pd.DataFrame.from_dict(metadata)

        # Save ordering of files in a new column 'memmap_idx'
        metadata['memmap_idx'] = np.arange(len(metadata))

        # Save csv for normal and abnormal images
        csv_dir = os.path.join('./', 'csvs', 'rsna')
        os.makedirs(csv_dir, exist_ok=True)
        normal = metadata[metadata.label == 0]
        print(f"Number of normal images: {len(normal)}")
        normal.to_csv(os.path.join(csv_dir, 'normal.csv'), index=True)

        abnormal = metadata[metadata.label != 0]
        print(f"Number of abnormal images: {len(abnormal)}")
        abnormal.to_csv(os.path.join(csv_dir, 'abnormal.csv'), index=True)

        # Write memmap files for whole dataset
        memmap_file = os.path.join(rsna_dir, 'memmap', 'data')
        os.makedirs(memmap_file, exist_ok=True)
        print(f"Writing memmap file '{memmap_file}'...")
        write_memmap(metadata.Path.values.tolist(), memmap_file,
            load_fn=partial(self.load_and_resize, target_size=(256, 256)), target_size=(256, 256))

    def load_and_resize(self, path: str, target_size: Tuple[int, int]):
        ds = dicom.dcmread(path)
        img = torch.tensor(ds.pixel_array, dtype=torch.float32)[None] / 255.
        img = T.CenterCrop(min(img.shape[1:]))(img)
        img = T.Resize(target_size, antialias=True)(img)
        return img