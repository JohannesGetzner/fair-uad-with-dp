import os
from glob import glob

import kaggle
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from src import RSNA_DIR

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS_MAPPING = {
    'Normal': 0,  # 8851, female: 2905, male: 4946, age mean: 44.94, std: 16.39, min: 2, max: 155
    'Lung Opacity': 1,  # 6012, female: 2502, male: 3510, age mean: 45.58, std: 17.46, min: 1, max: 92
    'No Lung Opacity / Not Normal': 2  # 11821, female: 5111, male: 6710, age mean: 49.33, std: 16.49, min: 1, max: 153
}

SEX_MAPPING = {
    'M': 0,
    'F': 1
}


def download_rsna(rsna_dir: str = RSNA_DIR):
    """Downloads the RSNA dataset."""
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'kmader/rsna-pneumonia-detection-challenge',
        path=rsna_dir,
        unzip=True
    )


def extract_metadata(rsna_dir: str = RSNA_DIR):
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
        metadata.append({
            'patientId': patient_id,
            'label': CLASS_MAPPING[label],
            'PatientAge': int(ds.PatientAge),
            'PatientSex': ds.PatientSex
        })

    metadata = pd.DataFrame.from_dict(metadata)
    metadata.to_csv(os.path.join(THIS_DIR, 'csvs', 'rsna_metadata.csv'), index=False)


def load_rsna_naive_split(rsna_dir: str = RSNA_DIR,
                          anomaly: str = 'lungOpacity'):
    assert anomaly in ['lungOpacity', 'otherAnomaly']
    """Naive train/val/test split."""
    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'csvs', 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    # Use 8051 normal samples for training
    train = normal_data.sample(n=8051, random_state=42)

    # Rest for validation and test
    rest_normal = normal_data[~normal_data.patientId.isin(train.patientId)]
    val_normal = rest_normal.sample(n=400, random_state=42)
    test_normal = rest_normal[~rest_normal.patientId.isin(val_normal.patientId)]
    val_test = data.sample(n=800, random_state=42)
    val = val_test.iloc[:400, :]
    test = val_test.iloc[400:, :]

    # Concatenate and shuffle
    val = pd.concat([val_normal, val]).sample(frac=1, random_state=42).reset_index(drop=True)
    test = pd.concat([test_normal, test]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}': val,
        f'test/{anomaly}': test,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros_like(data['PatientSex'].values)
    return filenames, labels, meta


def load_rsna_gender_split(rsna_dir: str = RSNA_DIR,
                           male_percent: float = 0.5,
                           anomaly: str = 'lungOpacity'):
    """Load data with age balanced val and test sets. Training is either young, avg, or old.
    lo = lung opacity
    oa = other anomaly
    """
    assert 0.0 <= male_percent <= 1.0
    assert anomaly in ['lungOpacity', 'otherAnomaly']
    female_percent = 1 - male_percent

    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'csvs', 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    normal_male = normal_data[normal_data.PatientSex == 'M']
    normal_female = normal_data[normal_data.PatientSex == 'F']
    male = data[data.PatientSex == 'M']
    female = data[data.PatientSex == 'F']

    # Save 100 male and 100 female samples for every label for validation and test
    # Normal
    val_test_normal_male = normal_male.sample(n=50, random_state=42)
    val_test_normal_female = normal_female.sample(n=50, random_state=42)
    val_test_male = male.sample(n=100, random_state=42)
    val_test_female = female.sample(n=100, random_state=42)
    val_male = val_test_male.iloc[:50, :]
    val_female = val_test_female.iloc[:50, :]
    test_male = val_test_male.iloc[50:, :]
    test_female = val_test_female.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_test_normal_male, val_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_female = pd.concat([val_test_normal_female, val_female]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_male = pd.concat([val_test_normal_male, test_male]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_female = pd.concat([val_test_normal_female, test_female]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.patientId.isin(val_test_normal_male.patientId)]
    rest_normal_female = normal_female[~normal_female.patientId.isin(val_test_normal_female.patientId)]
    n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=42)
    train_female = rest_normal_female.sample(n=n_female, random_state=42)

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Using {n_male} male and {n_female} female samples for training.")

    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}_male': val_male,
        f'val/{anomaly}_female': val_female,
        f'test/{anomaly}_male': test_male,
        f'test/{anomaly}_female': test_female,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([SEX_MAPPING[v] for v in data['PatientSex'].values])
    return filenames, labels, meta


def upsample_dataset(data:pd.DataFrame, upsampling_freq):
    """upsample a dataset to a given frequency."""
    pass




def load_rsna_age_two_split(rsna_dir: str = RSNA_DIR,
                            old_percent: float = 0.5,
                            anomaly: str = 'lungOpacity',
                            upsampling_strategy=None):
    """Load data with age balanced val and test sets. Training fraction of old
    and young patients can be specified.
    lo = lung opacity
    oa = other anomaly
    """
    assert 0.0 <= old_percent <= 1.0
    assert anomaly in ['lungOpacity', 'otherAnomaly']
    young_percent = 1 - old_percent

    # Load metadata
    metadata = pd.read_csv(os.path.join(THIS_DIR, 'csvs', 'rsna_metadata.csv'))
    normal_data = metadata[metadata.label == 0]
    if anomaly == 'lungOpacity':
        data = metadata[metadata.label == 1]
    else:
        data = metadata[metadata.label == 2]

    # Filter ages over 110 years (outliers)
    normal_data = normal_data[normal_data.PatientAge < 110]
    data = data[data.PatientAge < 110]

    # Split data into bins by age
    n_bins = 3
    t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]
    print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    normal_young = normal_data[normal_data.PatientAge < t[1]]
    normal_old = normal_data[normal_data.PatientAge >= t[2]]
    young = data[data.PatientAge < t[1]]
    old = data[data.PatientAge >= t[2]]

    # Save 100 young and 100 old samples for every label for validation and test
    # Normal
    # TODO: why 50 samples from the normal dataset only?
    # TODO: why 100 samples from the anomaly dataset?
    val_test_normal_young = normal_young.sample(n=50, random_state=42)
    val_test_normal_old = normal_old.sample(n=50, random_state=42)
    val_test_young = young.sample(n=100, random_state=42)
    val_test_old = old.sample(n=100, random_state=42)
    val_young = val_test_young.iloc[:50, :]
    val_old = val_test_old.iloc[:50, :]
    test_young = val_test_young.iloc[50:, :]
    test_old = val_test_old.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_young = pd.concat([val_test_normal_young, val_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_old = pd.concat([val_test_normal_old, val_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_young = pd.concat([val_test_normal_young, test_young]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_old = pd.concat([val_test_normal_old, test_old]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Rest for training
    rest_normal_young = normal_young[~normal_young.patientId.isin(val_test_normal_young.patientId)]
    rest_normal_old = normal_old[~normal_old.patientId.isin(val_test_normal_old.patientId)]
    n_samples = min(len(rest_normal_young), len(rest_normal_old))
    n_young = int(n_samples * young_percent)
    n_old = int(n_samples * old_percent)
    train_young = rest_normal_young.sample(n=n_young, random_state=42)
    train_old = rest_normal_old.sample(n=n_old, random_state=42)
    print(f"Using {len(train_young)} young and {len(train_old)} old samples for training.")
    if upsampling_strategy:
        if old_percent == 1 or young_percent == 1:
            raise ValueError("Cannot up-sample when one of the classes is 100%")
        num_add_samples = n_old - n_young
        if upsampling_strategy == "even":
            replication_factor = (n_young+num_add_samples)/n_young
            train_young_new = train_young.loc[train_young.index.repeat(np.floor(replication_factor))]
            if replication_factor % 1 != 0:
                num_remaining_replications = int(np.rint((replication_factor % 1)*n_young))
                additional_samples = train_young.sample(n=num_remaining_replications, replace=False, random_state=42)
                train_young_new = pd.concat([train_young_new, additional_samples])
            train_young = train_young_new
        else:
            train_young = pd.concat([train_young, train_young.sample(n=num_add_samples, replace=True, random_state=42)])
        print("Up-sampling young samples by", num_add_samples)
        print(f"Using {len(train_young)} young and {len(train_old)} old samples for training.")
    # Aggregate training set and shuffle
    train = pd.concat([train_young, train_old]).sample(frac=1, random_state=42).reset_index(drop=True)
    # Return
    filenames = {}
    labels = {}
    meta = {}
    sets = {
        'train': train,
        f'val/{anomaly}_young': val_young,
        f'val/{anomaly}_old': val_old,
        f'test/{anomaly}_young': test_young,
        f'test/{anomaly}_old': test_old,
    }
    img_dir = os.path.join(rsna_dir, 'stage_2_train_images')
    for mode, data in sets.items():
        filenames[mode] = [f'{img_dir}/{patient_id}.dcm' for patient_id in data.patientId]
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.where(data['PatientAge'] < t[1], 1, np.where(data['PatientAge'] >= t[2], 0, None))
    return filenames, labels, meta





if __name__ == '__main__':
    download_rsna()
    extract_metadata()
    pass
