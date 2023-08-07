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
#$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.0 --d "${date}"
#$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.25 --d "${date}"
#$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.50 --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --custom 0.9 --job_type_mod "lw09" --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --custom 0.7 --job_type_mod "lw07" --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --custom 0.5 --job_type_mod "lw05" --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --custom 0.3 --job_type_mod "lw03" --d "${date}"
$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 0.75 --custom 0.1 --job_type_mod "lw01" --d "${date}"
#$ANACONDA_PYTHON train.py --run_name dp_1 --protected_attr_percent 1.0 --d "${date}"
