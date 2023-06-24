#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')

python train.py --batch_size 16 --experiment_name "FAE-rsna-sweep-maxgradnorm-1-001-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50  --log_dir logs/max_grad_norm_sweep/old_percent_050 --dp --max_steps 60 --sweep
python train.py --batch_size 16 --experiment_name "FAE-rsna-sweep-maxgradnorm-1-001-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.50 --log_dir logs/max_grad_norm_sweep/male_percent_050 --dp --max_steps 60 --sweep
