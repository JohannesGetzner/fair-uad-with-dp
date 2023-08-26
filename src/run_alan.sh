#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --second_stage_epsilon -1 --d "${date}" --group_name_mod "bs32-ss"
python train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --second_stage_epsilon -1 --d "${date}" --group_name_mod "bs32-ss"
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_config normal --run_version v3 --protected_attr_percent 0.00 --d "${date}" --group_name_mod "bs64"
python train.py --run_config normal --run_version v3 --protected_attr_percent 0.25 --d "${date}" --group_name_mod "bs64"
python train.py --run_config normal --run_version v3 --protected_attr_percent 0.50 --d "${date}" --group_name_mod "bs64"
python train.py --run_config normal --run_version v3 --protected_attr_percent 0.75 --d "${date}" --group_name_mod "bs64"
python train.py --run_config normal --run_version v3 --protected_attr_percent 1.00 --d "${date}" --group_name_mod "bs64"