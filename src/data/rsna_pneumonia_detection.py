import os
from glob import glob

import kaggle
import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm
from src.data.data_utils import read_memmap, write_memmap
from src import RSNA_DIR, SEED
from functools import partial
from typing import Tuple
import torch
from torchvision import transforms as T

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
MAX_YOUNG = 31
MIN_OLD = 61

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


def load_rsna_gender_split(
        rsna_dir: str = RSNA_DIR,
        male_percent: float = 0.5,
        anomaly: str = 'lungOpacity',
        upsampling_strategy=None,
        effective_dataset_size=1.0,
        random_state=42
):
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
    val_test_normal_male = normal_male.sample(n=50, random_state=random_state)
    val_test_normal_female = normal_female.sample(n=50, random_state=random_state)
    val_test_male = male.sample(n=100, random_state=random_state)
    val_test_female = female.sample(n=100, random_state=random_state)
    val_male = val_test_male.iloc[:50, :]
    val_female = val_test_female.iloc[:50, :]
    test_male = val_test_male.iloc[50:, :]
    test_female = val_test_female.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_test_normal_male, val_male]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_female = pd.concat([val_test_normal_female, val_female]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_male = pd.concat([val_test_normal_male, test_male]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_female = pd.concat([val_test_normal_female, test_female]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.patientId.isin(val_test_normal_male.patientId)]
    rest_normal_female = normal_female[~normal_female.patientId.isin(val_test_normal_female.patientId)]
    n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=random_state)
    train_female = rest_normal_female.sample(n=n_female, random_state=random_state)
    if effective_dataset_size != 1.0:
        print(f"Reducing dataset size to {effective_dataset_size} ({len(train_female)} female and {len(train_male)} amle samples are available)")
        train_female = train_female.sample(frac=effective_dataset_size, random_state=random_state, replace=False)
        train_male = train_male.sample(frac=effective_dataset_size, random_state=random_state, replace=False)
    print(f"Using {len(train_female)} female and {len(train_male)} male samples for training.")
    if upsampling_strategy:
        if male_percent == 1 or female_percent == 1:
            raise ValueError("Cannot up-sample when one of the classes is 100%")
        num_add_samples = abs(n_male - n_female)
        if upsampling_strategy.endswith("female"):
            train_female = upsample_dataset(train_female, upsampling_strategy, num_add_samples)
        else:
            train_male = upsample_dataset(train_male, upsampling_strategy, num_add_samples)
        print(f"Using {len(train_female)} female and {len(train_male)} male samples for training.")

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("Final dataset shape: ", train.shape)

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


def load_rsna_age_two_split(
        rsna_dir: str = RSNA_DIR,
        old_percent: float = 0.5,
        anomaly: str = 'lungOpacity',
        upsampling_strategy=None,
        effective_dataset_size=1.0,
        random_state=42
):
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
    #n_bins = 3
    #t = np.histogram(normal_data.PatientAge, bins=n_bins)[1]
    print(f"Splitting data into bins by age: young < {MAX_YOUNG}, old >= {MIN_OLD}")

    normal_young = normal_data[normal_data.PatientAge < MIN_OLD]
    normal_old = normal_data[normal_data.PatientAge >= MAX_YOUNG]
    young = data[data.PatientAge < MIN_OLD]
    old = data[data.PatientAge >= MAX_YOUNG]

    # Save 100 young and 100 old samples for every label for validation and test
    # Normal
    val_test_normal_young = normal_young.sample(n=50, random_state=random_state)
    val_test_normal_old = normal_old.sample(n=50, random_state=random_state)
    val_test_young = young.sample(n=100, random_state=random_state)
    val_test_old = old.sample(n=100, random_state=random_state)
    val_young = val_test_young.iloc[:50, :]
    val_old = val_test_old.iloc[:50, :]
    test_young = val_test_young.iloc[50:, :]
    test_old = val_test_old.iloc[50:, :]
    # Aggregate validation and test sets and shuffle
    val_young = pd.concat([val_test_normal_young, val_young]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_old = pd.concat([val_test_normal_old, val_old]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_young = pd.concat([val_test_normal_young, test_young]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_old = pd.concat([val_test_normal_old, test_old]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Rest for training
    rest_normal_young = normal_young[~normal_young.patientId.isin(val_test_normal_young.patientId)]
    rest_normal_old = normal_old[~normal_old.patientId.isin(val_test_normal_old.patientId)]
    n_samples = min(len(rest_normal_young), len(rest_normal_old))
    n_young = int(n_samples * young_percent)
    n_old = int(n_samples * old_percent)
    train_young = rest_normal_young.sample(n=n_young, random_state=random_state)
    train_old = rest_normal_old.sample(n=n_old, random_state=random_state)
    if effective_dataset_size != 1.0:
        print(f"Reducing dataset size to {effective_dataset_size} ({len(train_young)} young and {len(train_old)} old samples are available)")
        train_young = train_young.sample(frac=effective_dataset_size, random_state=random_state, replace=False)
        train_old = train_old.sample(frac=effective_dataset_size, random_state=random_state, replace=False)
    print(f"Using {len(train_young)} young and {len(train_old)} old samples for training.")
    if upsampling_strategy:
        if old_percent == 1 or young_percent == 1:
            raise ValueError("Cannot up-sample when one of the classes is 100%")
        num_add_samples = abs(n_old - n_young)
        if upsampling_strategy.endswith("young"):
            train_young = upsample_dataset(train_young, upsampling_strategy, num_add_samples)
        else:
            train_old = upsample_dataset(train_old, upsampling_strategy, num_add_samples)
        print(f"Using {len(train_young)} young and {len(train_old)} old samples for training.")
    # Aggregate training set and shuffle
    train = pd.concat([train_young, train_old]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("Final dataset shape: ", train.shape)
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
        meta[mode] = np.where(data['PatientAge'] < MAX_YOUNG, 1, np.where(data['PatientAge'] >= MIN_OLD, 0, None))
    return filenames, labels, meta

def load_rsna_intersectional_age_sex_split(rsna_dir: str = RSNA_DIR):
    """Load MIMIC-CXR dataset with intersectional val and test sets."""
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'rsna')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into sets
    normal_male_young = normal[(normal.PatientSex == 'M') & (normal.PatientAge <= MAX_YOUNG)]
    normal_female_young = normal[(normal.PatientSex == 'F') & (normal.PatientAge <= MAX_YOUNG)]
    normal_male_old = normal[(normal.PatientSex == 'M') & (normal.PatientAge >= MIN_OLD)]
    normal_female_old = normal[(normal.PatientSex == 'F') & (normal.PatientAge >= MIN_OLD)]

    val_test_normal_male_young = normal_male_young.sample(n=100, random_state=SEED)
    val_test_normal_female_young = normal_female_young.sample(n=100, random_state=SEED)
    val_test_normal_male_old = normal_male_old.sample(n=100, random_state=SEED)
    val_test_normal_female_old = normal_female_old.sample(n=100, random_state=SEED)

    val_normal_male_young = val_test_normal_male_young[:50]
    val_normal_female_young = val_test_normal_female_young[:50]
    val_normal_male_old = val_test_normal_male_old[:50]
    val_normal_female_old = val_test_normal_female_old[:50]

    test_normal_male_young = val_test_normal_male_young[50:]
    test_normal_female_young = val_test_normal_female_young[50:]
    test_normal_male_old = val_test_normal_male_old[50:]
    test_normal_female_old = val_test_normal_female_old[50:]

    # Split abnormal images into sets
    abnormal_male_young = abnormal[(abnormal.PatientSex == 'M') & (abnormal.PatientAge <= MAX_YOUNG)]
    abnormal_female_young = abnormal[(abnormal.PatientSex == 'F') & (abnormal.PatientAge <= MAX_YOUNG)]
    abnormal_male_old = abnormal[(abnormal.PatientSex == 'M') & (abnormal.PatientAge >= MIN_OLD)]
    abnormal_female_old = abnormal[(abnormal.PatientSex == 'F') & (abnormal.PatientAge >= MIN_OLD)]

    val_test_abnormal_male_young = abnormal_male_young.sample(n=100, random_state=SEED)
    val_test_abnormal_female_young = abnormal_female_young.sample(n=100, random_state=SEED)
    val_test_abnormal_male_old = abnormal_male_old.sample(n=100, random_state=SEED)
    val_test_abnormal_female_old = abnormal_female_old.sample(n=100, random_state=SEED)

    val_abnormal_male_young = val_test_abnormal_male_young[:50]
    val_abnormal_female_young = val_test_abnormal_female_young[:50]
    val_abnormal_male_old = val_test_abnormal_male_old[:50]
    val_abnormal_female_old = val_test_abnormal_female_old[:50]

    test_abnormal_male_young = val_test_abnormal_male_young[50:]
    test_abnormal_female_young = val_test_abnormal_female_young[50:]
    test_abnormal_male_old = val_test_abnormal_male_old[50:]
    test_abnormal_female_old = val_test_abnormal_female_old[50:]

    # Merge and shuffle normal and abnormal val and test sets
    val_male_young = pd.concat([val_normal_male_young, val_abnormal_male_young]).sample(frac=1, random_state=SEED)
    val_female_young = pd.concat([val_normal_female_young, val_abnormal_female_young]).sample(frac=1, random_state=SEED)
    val_male_old = pd.concat([val_normal_male_old, val_abnormal_male_old]).sample(frac=1, random_state=SEED)
    val_female_old = pd.concat([val_normal_female_old, val_abnormal_female_old]).sample(frac=1, random_state=SEED)

    val_male = pd.concat([val_male_young, val_male_old]).sample(frac=1, random_state=SEED)
    val_female = pd.concat([val_female_young, val_female_old]).sample(frac=1, random_state=SEED)
    val_young = pd.concat([val_male_young, val_female_young]).sample(frac=1, random_state=SEED)
    val_old = pd.concat([val_male_old, val_female_old]).sample(frac=1, random_state=SEED)

    test_male_young = pd.concat([test_normal_male_young, test_abnormal_male_young]).sample(frac=1, random_state=SEED)
    test_female_young = pd.concat([test_normal_female_young, test_abnormal_female_young]).sample(frac=1, random_state=SEED)
    test_male_old = pd.concat([test_normal_male_old, test_abnormal_male_old]).sample(frac=1, random_state=SEED)
    test_female_old = pd.concat([test_normal_female_old, test_abnormal_female_old]).sample(frac=1, random_state=SEED)

    test_male = pd.concat([test_male_young, test_male_old]).sample(frac=1, random_state=SEED)
    test_female = pd.concat([test_female_young, test_female_old]).sample(frac=1, random_state=SEED)
    test_young = pd.concat([test_male_young, test_female_young]).sample(frac=1, random_state=SEED)
    test_old = pd.concat([test_male_old, test_female_old]).sample(frac=1, random_state=SEED)

    # Use rest of normal samples for training
    val_test_normal = pd.concat([
        val_test_normal_male_young,
        val_test_normal_female_young,
        val_test_normal_male_old,
        val_test_normal_female_old
    ])
    train = normal[~normal.patientId.isin(val_test_normal.patientId)]
    print(f"Using {len(train)} normal samples for training.")
    print(f"Average age of training samples: {train.PatientAge.mean():.2f}, std: {train.PatientAge.std():.2f}")
    print(f"Fraction of female samples in training: {(train.PatientSex == 'F').mean():.2f}")
    print(f"Fraction of male samples in training: {(train.PatientSex == 'M').mean():.2f}")
    print(f"Fraction of young samples in training: {(train.PatientAge <= MAX_YOUNG).mean():.2f}")
    print(f"Fraction of old samples in training: {(train.PatientAge >= MIN_OLD).mean():.2f}")

    print(f"val_male - samples: {len(val_male)}, male: {val_male.PatientSex.eq('M').mean():.2f}, female: {val_male.PatientSex.eq('F').mean():.2f}, young: {(val_male.PatientAge <= MAX_YOUNG).mean():.2f}, old {(val_male.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(val_male.label != 0).mean():.2f}")
    print(f"val_female - samples: {len(val_female)}, male: {val_female.PatientSex.eq('M').mean():.2f}, female: {val_female.PatientSex.eq('F').mean():.2f}, young: {(val_female.PatientAge <= MAX_YOUNG).mean():.2f}, old {(val_female.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(val_female.label != 0).mean():.2f}")
    print(f"val_young - samples: {len(val_young)}, male: {val_young.PatientSex.eq('M').mean():.2f}, female: {val_young.PatientSex.eq('F').mean():.2f}, young: {(val_young.PatientAge <= MAX_YOUNG).mean():.2f}, old {(val_young.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(val_young.label != 0).mean():.2f}")
    print(f"val_old - samples: {len(val_old)}, male: {val_old.PatientSex.eq('M').mean():.2f}, female: {val_old.PatientSex.eq('F').mean():.2f}, young: {(val_old.PatientAge <= MAX_YOUNG).mean():.2f}, old {(val_old.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(val_old.label != 0).mean():.2f}")
    print(f"test_male - samples: {len(test_male)}, male: {test_male.PatientSex.eq('M').mean():.2f}, female: {test_male.PatientSex.eq('F').mean():.2f}, young: {(test_male.PatientAge <= MAX_YOUNG).mean():.2f}, old {(test_male.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(test_male.label != 0).mean():.2f}")
    print(f"test_female - samples: {len(test_female)}, male: {test_female.PatientSex.eq('M').mean():.2f}, female: {test_female.PatientSex.eq('F').mean():.2f}, young: {(test_female.PatientAge <= MAX_YOUNG).mean():.2f}, old {(test_female.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(test_female.label != 0).mean():.2f}")
    print(f"test_young - samples: {len(test_young)}, male: {test_young.PatientSex.eq('M').mean():.2f}, female: {test_young.PatientSex.eq('F').mean():.2f}, young: {(test_young.PatientAge <= MAX_YOUNG).mean():.2f}, old {(test_young.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(test_young.label != 0).mean():.2f}")
    print(f"test_old - samples: {len(test_old)}, male: {test_old.PatientSex.eq('M').mean():.2f}, female: {test_old.PatientSex.eq('F').mean():.2f}, young: {(test_old.PatientAge <= MAX_YOUNG).mean():.2f}, old {(test_old.PatientAge >= MIN_OLD).mean():.2f}, anomalous: {(test_old.label != 0).mean():.2f}")

    img_data = read_memmap(
        os.path.join(
            rsna_dir,
            'memmap',
            'data'),
    )

    # Return
    actual_data = {}
    labels = {}
    meta = {}
    index_mapping = {}
    file_names = {}
    sets = {
        'train': train,
        'val/male': val_male,
        'val/female': val_female,
        'val/young': val_young,
        'val/old': val_old,
        'test/male': test_male,
        'test/female': test_female,
        'test/young': test_young,
        'test/old': test_old,
    }
    def get_meta_num(data):
        # young old combinations
        combos = {
            (True, True): 0,
            (True, False): 1,
            (False, True): 2,
            (False, False): 3
        }
        meta_mappings = data.apply(lambda x: combos[(x.PatientAge >= MIN_OLD, x.PatientSex == 'M')], axis=1)
        # return numpy array
        return meta_mappings.values

    for mode, data in sets.items():
        actual_data[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = get_meta_num(data)
        index_mapping[mode] = data.memmap_idx.values
        file_names[mode] = data.Path.values
    return actual_data, labels, meta, index_mapping, file_names

def upsample_dataset(data:pd.DataFrame, strategy:str, num_add_samples:int):
    n = len(data)
    if strategy.startswith("even"):
        replication_factor = (n + num_add_samples) / n
        data_new = data.loc[data.index.repeat(np.floor(replication_factor))]
        if replication_factor % 1 != 0:
            num_remaining_replications = int(np.rint((replication_factor % 1) * n))
            additional_samples = data.sample(n=num_remaining_replications, replace=False, random_state=42)
            data_new = pd.concat([data_new, additional_samples])
        data = data_new
    else:
        data = pd.concat([data, data.sample(n=num_add_samples, replace=True, random_state=42)])
    print("Up-sampling by", num_add_samples)
    return data


def prepare_rsna(rsna_dir: str = RSNA_DIR):
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
            'Path': file,
            'patientId': patient_id,
            'label': CLASS_MAPPING[label],
            'PatientAge': int(ds.PatientAge),
            'PatientSex': ds.PatientSex
        })

    metadata = pd.DataFrame.from_dict(metadata)

    # Save ordering of files in a new column 'memmap_idx'
    metadata['memmap_idx'] = np.arange(len(metadata))

    # Save csv for normal and abnormal images
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'rsna')
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
    write_memmap(
        metadata.Path.values.tolist(),
        memmap_file,
        load_fn=partial(load_and_resize, target_size=(256, 256)),
        target_size=(256, 256)
    )

def load_and_resize(path: str, target_size: Tuple[int, int]):
    ds = dicom.dcmread(path)
    img = torch.tensor(ds.pixel_array, dtype=torch.float32)[None] / 255.
    img = T.CenterCrop(min(img.shape[1:]))(img)
    img = T.Resize(target_size, antialias=True)(img)
    return img


if __name__ == '__main__':
    prepare_rsna()
    pass