#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')

python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --log_dir logs/max_grad_norm_sweep/old_percent_000 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --log_dir logs/max_grad_norm_sweep/old_percent_025 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50 --log_dir logs/max_grad_norm_sweep/old_percent_050 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.75 --log_dir logs/max_grad_norm_sweep/old_percent_075 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 1.00 --log_dir logs/max_grad_norm_sweep/old_percent_100 --dp --max_steps 8000 --sweep

python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --log_dir logs/max_grad_norm_sweep/male_percent_00 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --log_dir logs/max_grad_norm_sweep/male_percent_01 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.2 --log_dir logs/max_grad_norm_sweep/male_percent_02 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.3 --log_dir logs/max_grad_norm_sweep/male_percent_03 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.4 --log_dir logs/max_grad_norm_sweep/male_percent_04 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.5 --log_dir logs/max_grad_norm_sweep/male_percent_05 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.6 --log_dir logs/max_grad_norm_sweep/male_percent_06 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.7 --log_dir logs/max_grad_norm_sweep/male_percent_07 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.8 --log_dir logs/max_grad_norm_sweep/male_percent_08 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.9 --log_dir logs/max_grad_norm_sweep/male_percent_09 --dp --max_steps 8000 --sweep
python train.py --experiment_name "FAE-rsna-sex/age-max-gard-norm-sweep-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 1.0 --log_dir logs/max_grad_norm_sweep/male_percent_10 --dp --max_steps 8000 --sweep
