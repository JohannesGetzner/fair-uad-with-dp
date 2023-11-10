import os
import numpy as np
import pandas as pd
from .anomaly_dataset import AnomalyDataset, ATTRIBUTE_MAPPINGS
from . import RSNA_DIR


class RsnaAnomalyDataset(AnomalyDataset):
    def __init__(self, dataset_config):
        super().__init__(dataset_config)
        self.split_info = None

    def load_data(self, anomaly="lungOpacity"):
        metadata = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'csvs', 'rsna_metadata.csv'))
        normal_data = metadata[metadata.label == 0]
        if anomaly == 'lungOpacity':
            anomalous_data = metadata[metadata.label == 1]
        else:
            anomalous_data = metadata[metadata.label == 2]
        normal_data["id"] = normal_data[""]
        anomalous_data["id"] = anomalous_data[""]
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
        train = pd.concat(
            [*custom_data_loading_hook(train_A, train_B)]
        ).sample(frac=1, random_state=self.config["random_state"]).reset_index(drop=True)
        filenames = {}
        labels = {}
        meta = {}
        sets = {
            f'train': train,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': val_A,
            f'val/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': val_B,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["A"]}': test_A,
            f'test/{"lungOpacity"}_{ATTRIBUTE_MAPPINGS[self.config["protected_attr"]]["B"]}': test_B}
        img_dir = os.path.join(RSNA_DIR, 'stage_2_train_images')
        for mode, data in sets.items():
            filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
            labels[mode] = [min(1, label) for label in data.label.values]
            meta[mode] = self.encode_metadata(data)
        return self.construct_dataloaders(filenames, labels, meta)

    def prepare_dataset(self):
        # TODO: implement
        pass