import os
BASE_DIR = "/home/getznerj/Documents/Thesis/data"
RSNA_DIR = os.environ.get('RSNA_DIR', os.path.join(BASE_DIR, 'RSNA'))
CXR14_DIR = os.environ.get('CXR14_DIR', os.path.join(BASE_DIR,'CXR8'))
MIMIC_CXR_DIR = os.environ.get('MIMIC-CXR_DIR', os.path.join(BASE_DIR, 'MIMIC-CXR/mimic-cxr-jpg_2-0-0'))
CHEXPERT_DIR = os.environ.get('CHEXPERT_DIR', os.path.join(BASE_DIR, 'CheXpert'))