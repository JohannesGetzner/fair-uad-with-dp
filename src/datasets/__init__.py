import os
BASE_DIR = "/home/getznerj/Documents/Thesis/data"
RSNA_DIR = os.environ.get('RSNA_DIR', os.path.join(BASE_DIR, 'RSNA'))
CXR14_DIR = os.environ.get('CXR14_DIR', os.path.join(BASE_DIR,'CXR8'))

"""
Folder structure for datasets should be as follows:
- RSNA:
    - stage_2_train_images/
    - stage_2_test_images/
    - stage_2_train_labels.csv
    - stage_2_detailed_class_info.csv
    - memmap/

- CXR8:
    - images/
    - memmap/
    - Data_Entry_2017.csv
    
memmap directories are created by running the corresponding prepare_dataset() function. csvs are also created by running the prepare_dataset() function.
"""