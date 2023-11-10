import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RSNA_DIR = os.environ.get('RSNA_DIR', './datasets/RSNA')
CXR14_DIR = os.environ.get('CXR14_DIR', 'datasets/CXR8')
MIMIC_CXR_DIR = os.environ.get('MIMIC-CXR_DIR', '/vol/aimspace/projects/mimic_cxr/mimic-cxr-jpg_2-0-0')
CHEXPERT_DIR = os.environ.get('CHEXPERT_DIR', 'datasets/CheXpert')

SEED = 42

