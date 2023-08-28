#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --second_stage_epsilon -1 --second_stage_epochs 130 --d "${date}" --group_name_mod "bs32-ss"
python train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --second_stage_epsilon -1 --second_stage_epochs 130 --d "${date}" --group_name_mod "bs32-ss"
python train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --second_stage_epsilon -1 --second_stage_epochs 130 --d "${date}" --group_name_mod "bs32-ss"
python train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --second_stage_epsilon -1 --second_stage_epochs 130 --d "${date}" --group_name_mod "bs32-ss"
python train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --second_stage_epsilon -1 --second_stage_epochs 130 --d "${date}" --group_name_mod "bs32-ss"
