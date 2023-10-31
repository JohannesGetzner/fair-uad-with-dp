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
$ANACONDA_PYTHON -u train.py --run_config normal --run_version "balanced_baseline"               --hidden_dims 25 27 125 175                        --group_name_mod "bs32-balanced-ms25-27-125-175-baseline"        --d "${date}"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version "distill_baseline"   --no_img_log --hidden_dims 25 27 125 175 --n_training_samples 1 --group_name_mod "bs32-balanced-dataset-distillation-nsamples1"  --d "${date}"

date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version "balanced_baseline"               --hidden_dims 25 27 125 175                        --model_type "RD" --group_name_mod "bs32-balanced-ms25-27-125-175-baseline"         --d "${date}"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version "distill_baseline"   --no_img_log --hidden_dims 25 27 125 175 --n_training_samples 1 --model_type "RD" --group_name_mod "bs32-balanced-dataset-distillation-nsamples1"   --d "${date}"