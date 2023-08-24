#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_name non_dp --protected_attr_percent 0.5 --stage_two_epsilon 2 --d "${date}"
