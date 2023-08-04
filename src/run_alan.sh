#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_name non_dp --protected_attr_percent 0.0 --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.25 --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.5 --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 0.75 --d "${date}"
python train.py --run_name non_dp --protected_attr_percent 1.0 --d "${date}"
