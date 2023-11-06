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
print(os.getcwd())

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
    images = {}
    labels = {}
    meta = {}
    index_mapping = {}
    filenames = {}
    sets = {
        'train': train,
        'val/old': val_old,
        'val/young': val_young,
        'test/old': test_old,
        'test/young': test_young,
    }
    for mode, data in sets.items():
        images[mode] = memmap_file
        filenames[mode] = data.path.values
        labels[mode] = [min(1, label) for label in data.label.values]
        meta[mode] = np.where(data['Patient Age'] <= MAX_YOUNG, 1, np.where(data['Patient Age'] >= MIN_OLD, 0, None))
        #meta[mode] = np.zeros(len(data), dtype=np.float32)
        index_mapping[mode] = data.memmap_idx.values
    return images, labels, meta, index_mapping, filenames


def load_cxr14_intersectional_age_sex_split(cxr14_dir: str = CXR14_DIR):
    """Load MIMIC-CXR dataset with intersectional val and test sets."""
    csv_dir = os.path.join(THIS_DIR, 'csvs', 'cxr14_ap_pa')
    normal = pd.read_csv(os.path.join(csv_dir, 'normal.csv'))
    abnormal = pd.read_csv(os.path.join(csv_dir, 'abnormal.csv'))
    normal.rename({"Patient Age":"PatientAge", "Patient Gender":"PatientSex", "Patient ID": "patientId"}, axis=1, inplace=True)
    abnormal.rename({"Patient Age":"PatientAge", "Patient Gender":"PatientSex", "Patient ID": "patientId"}, axis=1, inplace=True)
    # UNCOMMENT TO ONLY USE PNEUMONIA IMAGES
    abnormal = abnormal[abnormal['Finding Labels'].str.contains("Pneumonia")]
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

    img_data = read_memmap(os.path.join(cxr14_dir, 'memmap', 'cxr14_ap_pa'), )

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
        file_names[mode] = data.path.values
    return actual_data, labels, meta, index_mapping, file_names


if __name__ == '__main__':

    prepare_cxr14()
    pass
