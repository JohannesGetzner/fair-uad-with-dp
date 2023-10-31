import os
RSNA_DIR = os.environ.get('RSNA_DIR', './datasets/RSNA')
CXR14_DIR = os.environ.get('CXR14_DIR', 'datasets/CXR8')
MIMIC_CXR_DIR = os.environ.get('MIMIC-CXR_DIR', 'datasets/MIMIC-CXR/mimic-cxr-jpg_2-0-0')
CHEXPERT_DIR = os.environ.get('CHEXPERT_DIR', 'datasets/CheXpert')