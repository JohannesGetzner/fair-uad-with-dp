#!/bin/bash
#SBATCH --partition=master
#SBATCH --ntasks=1
#SBATCH --time=07-00:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

date=$(date '+%Y-%m-%d %H:%M:%S')

# Load the Anaconda module
module load python/anaconda3

# Specify the Anaconda environment name
ANACONDA_ENV="fix"

# Specify the full path to the Anaconda environment's Python interpreter
ANACONDA_PYTHON="/u/home/getzner/.conda/envs/$ANACONDA_ENV/bin/python"

# Activate the Anaconda environment
source activate $ANACONDA_ENV

# Your commands using the Anaconda environment
echo "Running script with Anaconda environment: $ANACONDA_ENV"
#$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --weight 0.01 --job_type_mod "lw001" --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.25 --stage_two_epsilon 2 --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.5 --stage_two_epsilon 2 --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --stage_two_epsilon 2 --d "${date}"
#$ANACONDA_PYTHON sweep.py --num_sweeps 10 --protected_attr_percent 0.75 --weight 0.01 --job_type_mod "lw001" --d "${date}"
