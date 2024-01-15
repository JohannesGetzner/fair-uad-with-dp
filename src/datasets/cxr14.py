import os
import numpy as np
import pandas as pd
from PIL import Image
from functools import partial
from torchvision import transforms
from typing import Optional, Tuple
from .data_utils import read_memmap, write_memmap
from .anomaly_dataset import AnomalyDataset, ATTRIBUTE_MAPPINGS
from . import CXR14_DIR

MAX_YOUNG = 31
MIN_OLD = 61
CXR14LABELS = [  # All data
    'No Finding',  # 60361
    'Atelectasis',  # 11559
    'Cardiomegaly',  # 2776
    'Consolidation',  # 4667
    'Edema',  # 2303
    'Effusion',  # 13317
    'Emphysema',  # 2516
    'Fibrosis',  # 1686
    'Hernia',  # 227
    'Infiltration',  # 19894
    'Mass',  # 5782
    'Nodule',  # 6331
    'Pleural_Thickening',  # 3385
    'Pneumonia',  # 1431
    'Pneumothorax',  # 5302
]


class CXR14AnomalyDataset(AnomalyDataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.split_info = None

    def load_data(self, anomaly="lungOpacity"):
        csv_dir = os.path.join('datasets', 'csvs', 'cxr14_ap_pa')
        normal_data = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
        anomalous_data = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))
        normal_data["id"] = normal_data["Patient ID"]
        anomalous_data["id"] = anomalous_data["Patient ID"]
        return normal_data, anomalous_data

    def split_by_protected_attr(self, normal_data, anomalous_data):
        if self.config["protected_attr"] == "age":
            print(f"Splitting data by age. Below {MAX_YOUNG} is young, above {MIN_OLD} is old.")
            normal_data = normal_data[normal_data['Patient Age'] < 100]
            anomalous_data = anomalous_data[anomalous_data['Patient Age'] < 100]
            normal_young = normal_data[normal_data['Patient Age'] <= MAX_YOUNG]
            normal_old = normal_data[normal_data['Patient Age'] >= MIN_OLD]
            abnormal_young = anomalous_data[anomalous_data['Patient Age'] <= MAX_YOUNG]
            abnormal_old = anomalous_data[anomalous_data['Patient Age'] >= MIN_OLD]
            return normal_old, normal_young, abnormal_old, abnormal_young
        elif self.config["protected_attr"] == "sex":
            normal_male = normal_data[normal_data['Patient Gender'] == 'M']
            normal_female = normal_data[normal_data['Patient Gender'] == 'F']
            abnormal_male = anomalous_data[anomalous_data['Patient Gender'] == 'M']
            abnormal_female = anomalous_data[anomalous_data['Patient Gender'] == 'F']
            return normal_male, normal_female, abnormal_male, abnormal_female

    def get_dataloaders(self, custom_data_loading_hook):
        normal_data, anomalous_data = self.load_data()
        normal_A, normal_B, anomalous_A, anomalous_B = self.split_by_protected_attr(normal_data, anomalous_data)
        train_A, train_B, val_A, val_B, test_A, test_B = self.to_train_val_and_test(
            normal_A,
            normal_B,
            anomalous_A,
            anomalous_B,
            num_normal=1000,
            num_anomalous=1000
        )
        train = pd.concat([*custom_data_loading_hook(train_A, train_B)]).sample(frac=1, random_state=self.config[
            "random_state"]).reset_index(drop=True)

        memmap_file = read_memmap(os.path.join(CXR14_DIR, 'memmap', 'cxr14_ap_pa'), )
        images = {}
        labels = {}
        meta = {}
        index_mapping = {}
        filenames = {}
        sets = {
            'train': train,
            f'val/{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}':    val_A,
            f'val/{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}':    val_B,
            f'test/{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}':   test_A,
            f'test/{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}':   test_B,
        }
        for mode, data in sets.items():
            images[mode] = memmap_file
            filenames[mode] = data.path.values
            labels[mode] = [min(1, label) for label in data.label.values]
            meta[mode] = self.encode_metadata(data)
            index_mapping[mode] = data.memmap_idx.values
        return self.construct_dataloaders(images, labels, meta, filenames, index_mapping)

    def encode_metadata(self, data):
        if self.config["protected_attr"] == "age":
            return np.where(data['Patient Age'] <= MAX_YOUNG, 1, np.where(data['Patient Age'] >= MIN_OLD, 0, None))
        elif self.config["protected_attr"] == "sex":
            return np.array([0 if v == "M" else 1 for v in data['Patient Sex'].values])

    def prepare_dataset(self, cxr14_dir=CXR14_DIR):
        """Loads metadata (filenames, labels, age, gender) for each sample of the
        CXR14 dataset."""
        metadata = pd.read_csv(os.path.join(cxr14_dir, 'Data_Entry_2017.csv'))
        print(f"Total number of images: {len(metadata)}")

        # Prepend the path to the image filename
        metadata["path"] = metadata.apply(lambda row: os.path.join(cxr14_dir, "images", row['Image Index']), axis=1)

        # Reset index
        metadata = metadata.reset_index(drop=True)

        # Save ordering of files in a new column 'memmap_idx'
        metadata['memmap_idx'] = np.arange(len(metadata))

        memmap_dir = os.path.join(cxr14_dir, 'memmap')
        os.makedirs(memmap_dir, exist_ok=True)

        # csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_only')
        csv_dir = os.path.join("./", 'csvs', 'cxr14_ap_pa')
        os.makedirs(csv_dir, exist_ok=True)

        # Save csv for normal and abnormal images
        normal = metadata[metadata['Finding Labels'] == 'No Finding']
        print(f"Number of normal images: {len(normal)}")
        normal['label'] = [0] * len(normal)
        normal.to_csv(os.path.join(csv_dir, 'normal.csv'), index=True)

        abnormal = metadata[metadata['Finding Labels'] != 'No Finding']
        print(f"Number of abnormal images: {len(abnormal)}")
        abnormal['label'] = [1] * len(abnormal)
        abnormal.to_csv(os.path.join(csv_dir, 'abnormal.csv'), index=True)

        # Select sets of all pathologies
        pathologies = {}
        for i, pathology in enumerate(CXR14LABELS):
            # Filter all samples where pathology is in metadata['Finding Labels']
            pathologies[pathology] = metadata[metadata['Finding Labels'].str.contains(pathology)]
            print(f"Number of images for '{pathology}': {len(pathologies[pathology])}")

            # Add labels
            pathologies[pathology]['label'] = [i] * len(pathologies[pathology])

            # Save files
            pathologies[pathology].to_csv(os.path.join(csv_dir, f'{pathology}.csv'), index=True)

        # Write memmap files for whole dataset
        memmap_file = os.path.join(memmap_dir, 'cxr14_ap_pa')
        print(f"Writing memmap file '{memmap_file}'...")
        write_memmap(metadata['path'].values.tolist(), memmap_file,
            load_fn=partial(self.load_and_resize, target_size=(256, 256)), target_size=(256, 256))

    def load_and_resize(path: str, target_size: Tuple[int, int]):
        image = Image.open(path).convert('L')
        image = transforms.CenterCrop(min(image.size))(image)
        image = transforms.Resize(target_size)(image)
        image = transforms.ToTensor()(image)
        return image


if __name__ == "__main__":
    dataset = CXR14AnomalyDataset(dataset_config=None)
    dataset.prepare_dataset()