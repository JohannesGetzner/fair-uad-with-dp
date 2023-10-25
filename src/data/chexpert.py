"""Prepare CheXpert dataset for anomaly detection.

We only consider frontal view images.
Images with label "No Finding" are considered normal, all others anomal.
We exlude images with label "Uncertain" or "Support Devices".

CheXpert has the following structure:
train:
    patient<subject_id>:
        study<study_id>:
            view<view_id>_<frontal_or_lateral>.jpg
            ...
        ...
    ...
valid:
    ...
train.csv
valid.csv
"""
import os
from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.append('../')
from thesis.src import CHEXPERT_DIR, SEED
from thesis.src.data.data_utils import read_memmap, write_memmap


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEX_MAPPING = {
    'Male': 0,
    'Female': 1
}
RACE_MAPPING = {
    'Black': 0,
    'White': 1
}

MAX_YOUNG = 31  # 31  # 41
MIN_OLD = 61  # 61  # 66


CHEXPERT_LABELS = [
    'No Finding',
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
]


def load_and_resize(path: str, target_size: Tuple[int, int]):
    image = Image.open(path).convert('L')
    image = transforms.CenterCrop(min(image.size))(image)
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    return image


def prepare_chexpert(chexpert_dir: str = CHEXPERT_DIR):
    meta_train = pd.read_csv(os.path.join(chexpert_dir, 'train.csv'))
    meta_val = pd.read_csv(os.path.join(chexpert_dir, 'valid.csv'))
    metadata = pd.concat([meta_train, meta_val], ignore_index=True)
    metadata['subject_id'] = metadata.Path.apply(lambda path: path.split('/')[-3])
    print(f"Total number of images: {len(metadata)}")

    # Add race information
    df_demo = pd.DataFrame(pd.read_excel(os.path.join(chexpert_dir, 'CHEXPERT DEMO.xlsx'), engine='openpyxl'))
    df_demo = df_demo.rename(columns={'PATIENT': 'subject_id'})
    df_demo = df_demo.rename(columns={'PRIMARY_RACE': 'race'})
    df_demo = df_demo.rename(columns={'ETHNICITY': 'ethnicity'})
    df_demo = df_demo.drop(['GENDER', 'AGE_AT_CXR'], axis=1)
    metadata = pd.merge(metadata, df_demo, on='subject_id', how='left')

    # We only consider frontal view images. (AP and PA)
    metadata = metadata[metadata['Frontal/Lateral'] == 'Frontal']
    print(f"Number of frontal view images: {len(metadata)}")

    # Exclude images with support devices. 'Support Devices' is NaN
    metadata = metadata[metadata['Support Devices'].isna()]
    print(f"Number of images without support devices: {len(metadata)}")

    # Exclude images with uncertain labels. 'Uncertain' means no 1.0 or 0.0 in any label
    metadata = metadata[metadata[CHEXPERT_LABELS].isin([0.0, 1.0]).any(axis=1)]
    metadata[CHEXPERT_LABELS] = metadata[CHEXPERT_LABELS].replace(-1.0, float('nan'))
    print(f"Number of images with certain labels: {len(metadata)}\n")

    # Remove outlier with age 0
    metadata = metadata[metadata['Age'] > 0]

    # Add absolute path to images
    metadata.Path = metadata.apply(
        lambda row: os.path.join(
            chexpert_dir,
            '/'.join(row.Path.split('/')[1:])
        ), axis=1
    )

    # Reset index
    metadata = metadata.reset_index(drop=True)

    # Save ordering of files in a new column 'memmap_idx'
    metadata['memmap_idx'] = np.arange(len(metadata))

    csv_dir = os.path.join(THIS_DIR, 'csvs', 'chexpert_frontal')
    os.makedirs(csv_dir, exist_ok=True)

    # Save csv for normal and abnormal images
    normal = metadata[metadata['No Finding'] == 1.0]
    print(f"Number of normal images: {len(normal)}")
    normal['label'] = 0
    normal.to_csv(os.path.join(csv_dir, 'normal.csv'), index=True)

    abnormal = metadata[metadata['No Finding'].isna()]
    print(f"Number of abnormal images: {len(abnormal)}")
    abnormal['label'] = 1
    abnormal.to_csv(os.path.join(csv_dir, 'abnormal.csv'), index=True)

    # Write memmap files for whole dataset
    memmap_dir = os.path.join(chexpert_dir, 'memmap')
    os.makedirs(memmap_dir, exist_ok=True)
    memmap_file = os.path.join(memmap_dir,
                               'frontal_no_support_devices_no_uncertain',
                               'data')
    print(f"Writing memmap file '{memmap_file}'...")
    write_memmap(
        metadata.Path.values.tolist(),
        memmap_file,
        load_fn=partial(load_and_resize, target_size=(256, 256)),
        target_size=(256, 256)
    )


def load_chexpert_naive_split(chexpert_dir: str = CHEXPERT_DIR,
                              max_train_samples: Optional[int] = None):
    """Load MIMIC-CXR dataset with naive split."""
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'chexpert_frontal')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 1000 for val and test)
    normal_val_test = normal.sample(n=2000, random_state=SEED)
    normal_train = normal[~normal.subject_id.isin(normal_val_test.subject_id)]
    normal_train = normal_train.sample(n=max_train_samples, random_state=SEED) if max_train_samples else normal_train
    normal_val = normal_val_test[:1000]
    normal_test = normal_val_test[1000:]

    # Split abnormal images into val, test (use maximum 1000 for val and test)
    val = {}
    test = {}
    val_labels = {}
    test_labels = {}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        n_files = len(abnormal[pathology])
        n_use = min(n_files, 2000) // 2
        # Select anomal samples for val and test
        abnormal_val_test_pathology = abnormal[pathology].sample(n=2 * n_use, random_state=SEED)
        abnormal_val_pathology = abnormal_val_test_pathology[:n_use]
        abnormal_test_pathology = abnormal_val_test_pathology[n_use:]
        # Select normal samples for val and test
        normal_val_pathology = normal_val[:n_use]
        normal_test_pathology = normal_test[:n_use]
        # Merge normal and abnormal samples
        val[pathology] = pd.concat([abnormal_val_pathology, normal_val_pathology])
        test[pathology] = pd.concat([abnormal_test_pathology, normal_test_pathology])
        # Add labels
        val[pathology]['label'] = np.concatenate([np.ones(n_use), np.zeros(n_use)])
        test[pathology]['label'] = np.concatenate([np.ones(n_use), np.zeros(n_use)])
        # Shuffle
        val[pathology] = val[pathology].sample(frac=1, random_state=SEED)
        test[pathology] = test[pathology].sample(frac=1, random_state=SEED)
        # Save labels
        val_labels[pathology] = val[pathology]['label'].values
        test_labels[pathology] = test[pathology]['label'].values

    img_data = read_memmap(
        os.path.join(
            chexpert_dir,
            'memmap',
            'frontal_no_support_devices_no_uncertain'),
    )

    # Return
    filenames = {'train': img_data}
    labels = {'train': np.zeros(len(normal_train))}
    meta = {'train': np.zeros(len(normal_train))}
    index_mapping = {'train': normal_train['memmap_idx'].values}
    for pathology in CHEXPERT_LABELS:
        if pathology == 'No Finding':
            continue
        filenames[f'val/{pathology}'] = img_data
        labels[f'val/{pathology}'] = val_labels[pathology]
        meta[f'val/{pathology}'] = np.zeros(len(val[pathology]))
        index_mapping[f'val/{pathology}'] = val[pathology]['memmap_idx'].values
        filenames[f'test/{pathology}'] = img_data
        labels[f'test/{pathology}'] = test_labels[pathology]
        meta[f'test/{pathology}'] = np.zeros(len(test[pathology]))
        index_mapping[f'test/{pathology}'] = test[pathology]['memmap_idx'].values
    return filenames, labels, meta, index_mapping


def load_chexpert_sex_split(chexpert_dir: str = CHEXPERT_DIR,
                            male_percent: float = 0.5,
                            max_train_samples: Optional[int] = None):
    """Load data with sex-balanced val and test sets."""
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'chexpert_frontal')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 250 for val and test)
    normal_male = normal[normal.Sex == 'Male']
    normal_female = normal[normal.Sex == 'Female']
    val_test_normal_male = normal_male.sample(n=500, random_state=SEED)
    val_test_normal_female = normal_female.sample(n=500, random_state=SEED)
    val_normal_male = val_test_normal_male[:250]
    val_normal_female = val_test_normal_female[:250]
    test_normal_male = val_test_normal_male[250:]
    test_normal_female = val_test_normal_female[250:]

    # Split abnormal images into val, test (use maximum 250 for val and test)
    abnormal_male = abnormal[abnormal.Sex == 'Male']
    abnormal_female = abnormal[abnormal.Sex == 'Female']
    val_test_abnormal_male = abnormal_male.sample(n=500, random_state=SEED)
    val_test_abnormal_female = abnormal_female.sample(n=500, random_state=SEED)
    val_abnormal_male = val_test_abnormal_male.iloc[:250, :]
    val_abnormal_female = val_test_abnormal_female.iloc[:250, :]
    test_abnormal_male = val_test_abnormal_male.iloc[250:, :]
    test_abnormal_female = val_test_abnormal_female.iloc[250:, :]

    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_normal_male, val_abnormal_male]).sample(frac=1, random_state=SEED)
    val_female = pd.concat([val_normal_female, val_abnormal_female]).sample(frac=1, random_state=SEED)
    test_male = pd.concat([test_normal_male, test_abnormal_male]).sample(frac=1, random_state=SEED)
    test_female = pd.concat([test_normal_female, test_abnormal_female]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_male = normal_male[~normal_male.subject_id.isin(val_test_normal_male.subject_id)]
    rest_normal_female = normal_female[~normal_female.subject_id.isin(val_test_normal_female.subject_id)]
    if max_train_samples is not None:
        max_available = min(len(rest_normal_male), len(rest_normal_female)) * 2
        n_samples = min(max_available, max_train_samples)
    else:
        n_samples = min(len(rest_normal_male), len(rest_normal_female))
    n_male = int(n_samples * male_percent)
    n_female = int(n_samples * female_percent)
    train_male = rest_normal_male.sample(n=n_male, random_state=SEED)
    train_female = rest_normal_female.sample(n=n_female, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_male, train_female]).sample(frac=1, random_state=SEED)
    print(f"Using {n_male} male and {n_female} female samples for training.")

    img_data = read_memmap(
        os.path.join(
            chexpert_dir,
            'memmap',
            'frontal_no_support_devices_no_uncertain',
            'data'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/male': val_male,
        'val/female': val_female,
        'test/male': test_male,
        'test/female': test_female,
    }
    for mode, data in sets.items():
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([SEX_MAPPING[v] for v in data.Sex.values])
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_chexpert_age_split(chexpert_dir: str = CHEXPERT_DIR,
                            old_percent: float = 0.5,
                            max_train_samples: Optional[int] = None):
    """Load data with age-balanced val and test sets."""
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'chexpert_frontal')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split data into bins by age
    # n_bins = 3
    # t = np.histogram(normal.Age, bins=n_bins)[1]
    # print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    # normal_young = normal[normal.Age < t[1]]
    # normal_old = normal[normal.Age >= t[2]]
    # abnormal_young = abnormal[abnormal.Age < t[1]]
    # abnormal_old = abnormal[abnormal.Age >= t[2]]

    normal_young = normal[normal.Age <= MAX_YOUNG]
    normal_old = normal[normal.Age >= MIN_OLD]
    abnormal_young = abnormal[abnormal.Age <= MAX_YOUNG]
    abnormal_old = abnormal[abnormal.Age >= MIN_OLD]

    # Split normal images into train, val, test (use 250 for val and test)
    val_test_normal_old = normal_old.sample(n=500, random_state=SEED)
    val_test_normal_young = normal_young.sample(n=500, random_state=SEED)
    val_normal_old = val_test_normal_old[:250]
    val_normal_young = val_test_normal_young[:250]
    test_normal_old = val_test_normal_old[250:]
    test_normal_young = val_test_normal_young[250:]

    # Split abnormal images into val, test (use maximum 250 for val and test)
    val_test_abnormal_old = abnormal_old.sample(n=500, random_state=SEED)
    val_test_abnormal_young = abnormal_young.sample(n=500, random_state=SEED)
    val_abnormal_old = val_test_abnormal_old.iloc[:250, :]
    val_abnormal_young = val_test_abnormal_young.iloc[:250, :]
    test_abnormal_old = val_test_abnormal_old.iloc[250:, :]
    test_abnormal_young = val_test_abnormal_young.iloc[250:, :]

    # Aggregate validation and test sets and shuffle
    val_old = pd.concat([val_normal_old, val_abnormal_old]).sample(frac=1, random_state=SEED)
    val_young = pd.concat([val_normal_young, val_abnormal_young]).sample(frac=1, random_state=SEED)
    test_old = pd.concat([test_normal_old, test_abnormal_old]).sample(frac=1, random_state=SEED)
    test_young = pd.concat([test_normal_young, test_abnormal_young]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_old = normal_old[~normal_old.subject_id.isin(val_test_normal_old.subject_id)]
    rest_normal_young = normal_young[~normal_young.subject_id.isin(val_test_normal_young.subject_id)]
    if max_train_samples is not None:
        max_available = min(len(rest_normal_old), len(rest_normal_young)) * 2
        n_samples = min(max_available, max_train_samples)
    else:
        n_samples = min(len(rest_normal_old), len(rest_normal_young))
    n_old = int(n_samples * old_percent)
    n_young = int(n_samples * young_percent)
    train_old = rest_normal_old.sample(n=n_old, random_state=SEED)
    train_young = rest_normal_young.sample(n=n_young, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_old, train_young]).sample(frac=1, random_state=SEED)
    print(f"Using {n_old} old and {n_young} young samples for training.")

    img_data = read_memmap(
        os.path.join(
            chexpert_dir,
            'memmap',
            'frontal_no_support_devices_no_uncertain',
            'data'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/old': val_old,
        'val/young': val_young,
        'test/old': test_old,
        'test/young': test_young,
    }
    for mode, data in sets.items():
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)  # Unused
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_chexpert_race_split(chexpert_dir: str = CHEXPERT_DIR,
                             white_percent: float = 0.5,
                             max_train_samples: Optional[int] = None):
    """Load data with race-balanced val and test sets."""
    if white_percent is not None:
        assert 0.0 <= white_percent <= 1.0
        black_percent = 1 - white_percent

    # Load metadata
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'chexpert_frontal')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))
    print(len(normal), len(abnormal))

    # Only consider white and black patients
    mask = (normal.race.str.contains('Black', na=False))
    normal.loc[mask, 'race'] = 'Black'
    mask = (normal.race.str.contains('White', na=False))
    normal.loc[mask, 'race'] = 'White'
    mask = (abnormal.race.str.contains('Black', na=False))
    abnormal.loc[mask, 'race'] = 'Black'
    mask = (abnormal.race.str.contains('White', na=False))
    abnormal.loc[mask, 'race'] = 'White'
    normal = normal[normal.race.isin(['Black', 'White'])]
    abnormal = abnormal[abnormal.race.isin(['Black', 'White'])]

    n_v_t = 50

    # Split normal images into train, val, test
    normal_black = normal[normal.race == 'Black']
    normal_white = normal[normal.race == 'White']
    val_test_normal_black = normal_black.sample(n=2 * n_v_t, random_state=SEED)
    val_test_normal_white = normal_white.sample(n=2 * n_v_t, random_state=SEED)
    val_normal_black = val_test_normal_black[:n_v_t]
    val_normal_white = val_test_normal_white[:n_v_t]
    test_normal_black = val_test_normal_black[n_v_t:]
    test_normal_white = val_test_normal_white[n_v_t:]

    # Split abnormal images into val, test
    abnormal_black = abnormal[abnormal.race == 'Black']
    abnormal_white = abnormal[abnormal.race == 'White']
    val_test_abnormal_black = abnormal_black.sample(n=2 * n_v_t, random_state=SEED)
    val_test_abnormal_white = abnormal_white.sample(n=2 * n_v_t, random_state=SEED)
    val_abnormal_black = val_test_abnormal_black.iloc[:n_v_t, :]
    val_abnormal_white = val_test_abnormal_white.iloc[:n_v_t, :]
    test_abnormal_black = val_test_abnormal_black.iloc[n_v_t:, :]
    test_abnormal_white = val_test_abnormal_white.iloc[n_v_t:, :]

    # Aggregate validation and test sets and shuffle
    val_black = pd.concat([val_normal_black, val_abnormal_black]).sample(frac=1, random_state=SEED)
    val_white = pd.concat([val_normal_white, val_abnormal_white]).sample(frac=1, random_state=SEED)
    test_black = pd.concat([test_normal_black, test_abnormal_black]).sample(frac=1, random_state=SEED)
    test_white = pd.concat([test_normal_white, test_abnormal_white]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_black = normal_black[~normal_black.subject_id.isin(val_test_normal_black.subject_id)]
    rest_normal_white = normal_white[~normal_white.subject_id.isin(val_test_normal_white.subject_id)]
    if white_percent is None:
        n_white = len(rest_normal_white)
        n_black = len(rest_normal_black)
        if max_train_samples is not None:
            frac_white = n_white / (n_white + n_black)
            n_white = int(max_train_samples * frac_white)
            n_black = max_train_samples - n_white
    else:
        if max_train_samples is not None:
            max_available = min(len(rest_normal_black), len(rest_normal_white)) * 2
            n_samples = min(max_available, max_train_samples)
        else:
            n_samples = min(len(rest_normal_black), len(rest_normal_white))
        n_black = int(n_samples * black_percent)
        n_white = int(n_samples * white_percent)
    train_black = rest_normal_black.sample(n=n_black, random_state=SEED)
    train_white = rest_normal_white.sample(n=n_white, random_state=SEED)

    # Aggregate training set and shuffle
    train = pd.concat([train_black, train_white]).sample(frac=1, random_state=SEED)
    print(f"Using {n_black} black and {n_white} white samples for training.")

    img_data = read_memmap(
        os.path.join(
            chexpert_dir,
            'memmap',
            'frontal_no_support_devices_no_uncertain',
            'data'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val/black': val_black,
        'val/white': val_white,
        'test/black': test_black,
        'test/white': test_white,
    }
    for mode, data in sets.items():
        filenames[mode] = img_data
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.array([RACE_MAPPING[v] for v in data.race.values])
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


if __name__ == '__main__':
    prepare_chexpert()
    pass
