import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RSNA_DIR = os.environ.get('RSNA_DIR', './datasets/RSNA')
