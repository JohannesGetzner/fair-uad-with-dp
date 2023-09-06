#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')

python -u train.py --run_config local --run_version v1 --protected_attr_percent 0.25 --hidden_dims 25 75 125 175   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms25-75-125-175"
python -u train.py --run_config local --run_version v1 --protected_attr_percent 0.25 --hidden_dims 50 100 150 200  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms50-100-150-200"