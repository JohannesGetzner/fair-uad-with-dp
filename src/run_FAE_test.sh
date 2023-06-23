#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --batch_size 8 --model_type FAE --dataset rsna --protected_attr age --old_percent 0.5 --max_steps 200 --log_dir logs/test --experiment_name "FAE-rsna-sex/age-${date}" --dp --sweep
