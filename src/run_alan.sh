#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --run_name dp_1 --protected_attr_percent 1.0 --d "${date}"