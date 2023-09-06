#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')

python train.py --run_config local --run_version v1 --protected_attr_percent 0.0 --effective_dataset_size 0.7 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "01"
python train.py --run_config local --run_version v1 --protected_attr_percent 0.0 --effective_dataset_size 0.01 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "001"

python train.py --run_config local --run_version v1 --protected_attr_percent 0.25 --effective_dataset_size 0.7 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "07"
python train.py --run_config local --run_version v1 --protected_attr_percent 0.25 --effective_dataset_size 0.01 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "04"