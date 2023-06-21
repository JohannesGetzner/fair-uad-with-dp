#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_000 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_025 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_050 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.75 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_075 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 1.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_100 --max_steps 6000

python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_00 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_01 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.2 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_02 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.3 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_03 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.4 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_04 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.5 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_05 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.6 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_06 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.7 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_07 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.8 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_08 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.9 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_09 --max_steps 6000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 1.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_10 --max_steps 6000



python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_000 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_025 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_050 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 0.75 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_075 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr age --old_percent 1.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_100 --dp --max_steps 10000

python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_00 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_01 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.2 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_02 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.3 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_03 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.4 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_04 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.5 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_05 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.6 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_06 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.7 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_07 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.8 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_08 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.9 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_09 --dp --max_steps 10000
python train.py --experiment_name "FAE-rsna-sex/age-${date}" --model_type FAE --dataset rsna --protected_attr sex --male_percent 1.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_10 --dp --max_steps 10000
