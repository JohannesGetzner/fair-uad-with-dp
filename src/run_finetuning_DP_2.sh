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
# AGE DP
# ---------
#date=$(date '+%Y-%m-%d %H:%M:%S')
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1_finetuning --protected_attr_percent 0.00 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1_finetuning --protected_attr_percent 0.25 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1_finetuning --protected_attr_percent 0.50 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1_finetuning --protected_attr_percent 0.75 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
#$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1_finetuning --protected_attr_percent 1.00 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"

# ---------
# SEX DP
# ---------
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2_finetuning --protected_attr_percent 0.00 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2_finetuning --protected_attr_percent 0.25 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2_finetuning --protected_attr_percent 0.50 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2_finetuning --protected_attr_percent 0.75 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2_finetuning --protected_attr_percent 1.00 --second_stage_steps 5000 --second_stage_epsilon 2 --no_img_log --group_name_mod  "bs512-finetuning" --d "${date}"