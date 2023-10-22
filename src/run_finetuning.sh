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

# ---------
# AGE
# ---------
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.10 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.20 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.30 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.40 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.60 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.70 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.80 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.90 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
# ---------
# AGE DP
# ---------
#date=$(date '+%Y-%m-%d %H:%M:%S')
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.00 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 1.00 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
# ---------
# SEX
# ---------
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.10 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.20 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.30 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.40 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.60 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.70 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.80 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.90 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --second_stage_steps 2000 --group_name_mod  "bs32-finetuning" --d "${date}"
# ---------
# SEX DP
# ---------
#date=$(date '+%Y-%m-%d %H:%M:%S')
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.00 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 1.00 --second_stage_steps 11250 --second_stage_epsilon 3 --group_name_mod  "bs512-finetuning" --d "${date}"