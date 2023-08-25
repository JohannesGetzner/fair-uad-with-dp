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

$ANACONDA_PYTHON sweep.py --protected_attr_percent 0.75 --d "${date}" --group_name_mod "SWEEP-lr-bs1024-mgn001"
# $ANACONDA_PYTHON train.py --run_config dp --run_version v3 --protected_attr_percent 0.75 --n_adam --lr 0.002 --d "${date}" --group_name_mod "nadam-lr0002"
