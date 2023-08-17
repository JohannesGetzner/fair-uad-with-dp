#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.8 --job_type_mod "lw08" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.7 --job_type_mod "lw07" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.6 --job_type_mod "lw06" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.5 --job_type_mod "lw05" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.4 --job_type_mod "lw04" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.3 --job_type_mod "lw03" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.2 --job_type_mod "lw02" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.1 --job_type_mod "lw01" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.01 --job_type_mod "lw001" --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --weight 0.005 --job_type_mod "lw0005" --d "${date}"
