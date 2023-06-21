#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 3 --log_dir logs/FAE_rsna_age/old_percent_000 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}"
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 3 --log_dir logs/FAE_rsna_age/old_percent_025 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}"
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 3 --log_dir logs/FAE_rsna_age/old_percent_000 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}" --dp
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 3 --log_dir logs/FAE_rsna_age/old_percent_025 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}" --dp

python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 3 --log_dir logs/FAE_rsna_sex/male_percent_00 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}"
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 3 --log_dir logs/FAE_rsna_sex/male_percent_01 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}"
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 3 --log_dir logs/FAE_rsna_sex/male_percent_00 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}" --dp
python train.py --batch_size 24 --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 3 --log_dir logs/FAE_rsna_sex/male_percent_01 --max_steps 200 --experiment_name "FAE-rsna-sex/age-${date}" --dp