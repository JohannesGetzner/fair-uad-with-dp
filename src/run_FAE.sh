#!/bin/bash
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_000
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_025
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_050
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.75 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_075
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 1.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_100

python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_00
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_01
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.2 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_02
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.3 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_03
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.4 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_04
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.5 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_05
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.6 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_06
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.7 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_07
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.8 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_08
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.9 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_09
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 1.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_10



python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_000 --dp
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.25 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_025 --dp
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.50 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_050 --dp
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 0.75 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_075 --dp
python train.py --model_type FAE --dataset rsna --protected_attr age --old_percent 1.00 --num_seeds 5 --log_dir logs/FAE_rsna_age/old_percent_100 --dp

python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_00 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.1 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_01 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.2 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_02 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.3 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_03 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.4 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_04 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.5 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_05 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.6 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_06 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.7 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_07 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.8 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_08 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 0.9 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_09 --dp
python train.py --model_type FAE --dataset rsna --protected_attr sex --male_percent 1.0 --num_seeds 5 --log_dir logs/FAE_rsna_sex/male_percent_10 --dp