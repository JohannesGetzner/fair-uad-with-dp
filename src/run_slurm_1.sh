#!/bin/bash
#SBATCH --partition=master
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load the Anaconda module
module load python/anaconda3

# Specify the Anaconda environment name
ANACONDA_ENV="thesis_opacus"

# Specify the full path to the Anaconda environment's Python interpreter
ANACONDA_PYTHON="/u/home/getzner/.conda/envs/$ANACONDA_ENV/bin/python"

# Activate the Anaconda environment
source activate $ANACONDA_ENV

# Your commands using the Anaconda environment
echo "Running script with Anaconda environment: $ANACONDA_ENV"

date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 42 --group_name_mod "bs32-dss" --job_type_mod "0001-rs42" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 42 --group_name_mod "bs32-dss" --job_type_mod "0005-rs42" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 42 --group_name_mod "bs32-dss" --job_type_mod "0008-rs42" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 43 --group_name_mod "bs32-dss" --job_type_mod "0001-rs43" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 43 --group_name_mod "bs32-dss" --job_type_mod "0005-rs43" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 43 --group_name_mod "bs32-dss" --job_type_mod "0008-rs43" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 44 --group_name_mod "bs32-dss" --job_type_mod "0001-rs44" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 44 --group_name_mod "bs32-dss" --job_type_mod "0005-rs44" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 44 --group_name_mod "bs32-dss" --job_type_mod "0008-rs44" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 45 --group_name_mod "bs32-dss" --job_type_mod "0001-rs45" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 45 --group_name_mod "bs32-dss" --job_type_mod "0005-rs45" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 45 --group_name_mod "bs32-dss" --job_type_mod "0008-rs45" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 46 --group_name_mod "bs32-dss" --job_type_mod "0001-rs46" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 46 --group_name_mod "bs32-dss" --job_type_mod "0005-rs46" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 46 --group_name_mod "bs32-dss" --job_type_mod "0008-rs46" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --dataset_random_state 47 --group_name_mod "bs32-dss" --job_type_mod "0001-rs47" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --dataset_random_state 47 --group_name_mod "bs32-dss" --job_type_mod "0005-rs47" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 0.008 --dataset_random_state 47 --group_name_mod "bs32-dss" --job_type_mod "0008-rs47" --d "${date}"