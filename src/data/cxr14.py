"""
Split CXR14 dataset.
train_ad: 40000 no finding samples from train_val_list.txt
train_cls: 90% of the remaining samples from train_val_list.txt
val_cls: 10% of the remaining samples from train_val_list.txt
test_cls: all samples from test_list.txt
"""
import os
import sys
from pathlib import Path
#sys.path.insert(0, str(Path(__file__).parent.parent))  # root dir


from functools import partial
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

sys.path.append('../')
from src import CXR14_DIR, SEED
from src.data.data_utils import read_memmap, write_memmap

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SEX_MAPPING = {
    'M': 0,
    'F': 1
}

MAX_YOUNG = 31  # 31  # 41
MIN_OLD = 61  # 61  # 66

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


def prepare_cxr14(cxr14_dir: str = CXR14_DIR):
    """Loads metadata (filenames, labels, age, gender) for each sample of the
    CXR14 dataset."""
    metadata = pd.read_csv(os.path.join(cxr14_dir, 'Data_Entry_2017.csv'))
    print(f"Total number of images: {len(metadata)}")

    # Prepend the path to the image filename
    metadata["path"] = metadata.apply(
        lambda row: os.path.join(
            cxr14_dir,
            "images",
            row['Image Index']
        ), axis=1
    )

    # Reset index
    metadata = metadata.reset_index(drop=True)

    # Save ordering of files in a new column 'memmap_idx'
    metadata['memmap_idx'] = np.arange(len(metadata))

    memmap_dir = os.path.join(cxr14_dir, 'memmap')
    os.makedirs(memmap_dir, exist_ok=True)

    # csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_only')
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_pa')
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
    write_memmap(
        metadata['path'].values.tolist(),
        memmap_file,
        load_fn=partial(load_and_resize, target_size=(256, 256)),
        target_size=(256, 256)
    )


def load_and_resize(path: str, target_size: Tuple[int, int]):
    image = Image.open(path).convert('L')
    image = transforms.CenterCrop(min(image.size))(image)
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    return image


def load_cxr14_naive_split(cxr14_dir: str = CXR14_DIR,
                           max_train_samples: Optional[int] = None):
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 1000 for val and test)
    val_test_normal = normal.sample(n=2000, random_state=SEED)
    train = normal[~normal['Patient ID'].isin(val_test_normal['Patient ID'])]
    train = train.sample(n=max_train_samples, random_state=SEED) if max_train_samples else train
    val_normal = val_test_normal[:1000]
    test_normal = val_test_normal[1000:]

    # Split abnormal images into val, test (use maximum 1000 for val and test)
    val_test_abnormal = abnormal.sample(n=2000, random_state=SEED)
    val_abnormal = val_test_abnormal[:1000]
    test_abnormal = val_test_abnormal[1000:]

    # Aggregate validation and test sets and shuffle
    val = pd.concat([val_normal, val_abnormal]).sample(frac=1, random_state=SEED)
    test = pd.concat([test_normal, test_abnormal]).sample(frac=1, random_state=SEED)

    memmap_file = read_memmap(
        os.path.join(
            cxr14_dir,
            'memmap',
            'cxr14_ap_pa'),
    )

    # Return
    filenames = {}
    labels = {}
    meta = {}
    index_mapping = {}
    sets = {
        'train': train,
        'val': val,
        'test': test,
    }
    for mode, data in sets.items():
        filenames[mode] = memmap_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_cxr14_sex_split(cxr14_dir: str = CXR14_DIR,
                         male_percent: float = 0.5,
                         max_train_samples: Optional[int] = None):
    """Load data with sex-balanced val and test sets."""
    assert 0.0 <= male_percent <= 1.0
    female_percent = 1 - male_percent

    csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Split normal images into train, val, test (use 500 for val and test)
    normal_male = normal[normal['Patient Gender'] == 'M']
    normal_female = normal[normal['Patient Gender'] == 'F']
    val_test_normal_male = normal_male.sample(n=1000, random_state=SEED)
    val_test_normal_female = normal_female.sample(n=1000, random_state=SEED)
    val_normal_male = val_test_normal_male[:500]
    val_normal_female = val_test_normal_female[:500]
    test_normal_male = val_test_normal_male[500:]
    test_normal_female = val_test_normal_female[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    abnormal_male = abnormal[abnormal['Patient Gender'] == 'M']
    abnormal_female = abnormal[abnormal['Patient Gender'] == 'F']
    val_test_abnormal_male = abnormal_male.sample(n=1000, random_state=SEED)
    val_test_abnormal_female = abnormal_female.sample(n=1000, random_state=SEED)
    val_abnormal_male = val_test_abnormal_male[:500]
    val_abnormal_female = val_test_abnormal_female[:500]
    test_abnormal_male = val_test_abnormal_male[500:]
    test_abnormal_female = val_test_abnormal_female[500:]

    # Aggregate validation and test sets and shuffle
    val_male = pd.concat([val_normal_male, val_abnormal_male]).sample(frac=1, random_state=SEED)
    val_female = pd.concat([val_normal_female, val_abnormal_female]).sample(frac=1, random_state=SEED)
    test_male = pd.concat([test_normal_male, test_abnormal_male]).sample(frac=1, random_state=SEED)
    test_female = pd.concat([test_normal_female, test_abnormal_female]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_male = normal_male[~normal_male['Patient ID'].isin(val_test_normal_male['Patient ID'])]
    rest_normal_female = normal_female[~normal_female['Patient ID'].isin(val_test_normal_female['Patient ID'])]
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

    memmap_file = read_memmap(
        os.path.join(
            cxr14_dir,
            'memmap',
            'cxr14_ap_pa'),
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
        filenames[mode] = memmap_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


def load_cxr14_age_split(cxr14_dir: str = CXR14_DIR,
                         old_percent: float = 0.5,
                         max_train_samples: Optional[int] = None):
    """Load data with age-balanced val and test sets."""
    assert 0.0 <= old_percent <= 1.0
    young_percent = 1 - old_percent

    csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))

    # Filter ages over 100 years
    normal = normal[normal['Patient Age'] < 100]
    abnormal = abnormal[abnormal['Patient Age'] < 100]

    # Split data into bins by age
    # n_bins = 3
    # t = np.histogram(normal['Patient Age'], bins=n_bins)[1]
    # print(f"Splitting data into {n_bins - 1} bins by age: {t}")

    normal_young = normal[normal['Patient Age'] <= MAX_YOUNG]
    normal_old = normal[normal['Patient Age'] >= MIN_OLD]
    abnormal_young = abnormal[abnormal['Patient Age'] <= MAX_YOUNG]
    abnormal_old = abnormal[abnormal['Patient Age'] >= MIN_OLD]

    # Split normal images into train, val, test (use 500 for val and test)
    val_test_normal_old = normal_old.sample(n=1000, random_state=SEED)
    val_test_normal_young = normal_young.sample(n=1000, random_state=SEED)
    val_normal_old = val_test_normal_old[:500]
    val_normal_young = val_test_normal_young[:500]
    test_normal_old = val_test_normal_old[500:]
    test_normal_young = val_test_normal_young[500:]

    # Split abnormal images into val, test (use maximum 500 for val and test)
    val_test_abnormal_old = abnormal_old.sample(n=1000, random_state=SEED)
    val_test_abnormal_young = abnormal_young.sample(n=1000, random_state=SEED)
    val_abnormal_old = val_test_abnormal_old[:500]
    val_abnormal_young = val_test_abnormal_young[:500]
    test_abnormal_old = val_test_abnormal_old[500:]
    test_abnormal_young = val_test_abnormal_young[500:]

    # Aggregate validation and test sets and shuffle
    val_old = pd.concat([val_normal_old, val_abnormal_old]).sample(frac=1, random_state=SEED)
    val_young = pd.concat([val_normal_young, val_abnormal_young]).sample(frac=1, random_state=SEED)
    test_old = pd.concat([test_normal_old, test_abnormal_old]).sample(frac=1, random_state=SEED)
    test_young = pd.concat([test_normal_young, test_abnormal_young]).sample(frac=1, random_state=SEED)

    # Rest for training
    rest_normal_old = normal_old[~normal_old['Patient ID'].isin(val_test_normal_old['Patient ID'])]
    rest_normal_young = normal_young[~normal_young['Patient ID'].isin(val_test_normal_young['Patient ID'])]
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

    memmap_file = read_memmap(
        os.path.join(
            cxr14_dir,
            'memmap',
            'cxr14_ap_pa'),
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
        filenames[mode] = memmap_file
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.memmap_idx.values
    return filenames, labels, meta, index_mapping


if __name__ == '__main__':

    prepare_cxr14()
    pass
