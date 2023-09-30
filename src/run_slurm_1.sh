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

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --dataset_random_state 43 --group_name_mod "bs32-dss" --job_type_mod "1-rs43" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --dataset_random_state 44 --group_name_mod "bs32-dss" --job_type_mod "1-rs44" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --dataset_random_state 45 --group_name_mod "bs32-dss" --job_type_mod "1-rs45" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --dataset_random_state 46 --group_name_mod "bs32-dss" --job_type_mod "1-rs46" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --dataset_random_state 47 --group_name_mod "bs32-dss" --job_type_mod "1-rs47" --d "${date}"