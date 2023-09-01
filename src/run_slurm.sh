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
ANACONDA_ENV="thesis_opacus"

# Specify the full path to the Anaconda environment's Python interpreter
ANACONDA_PYTHON="/u/home/getzner/.conda/envs/$ANACONDA_ENV/bin/python"

# Activate the Anaconda environment
source activate $ANACONDA_ENV

# Your commands using the Anaconda environment
echo "Running script with Anaconda environment: $ANACONDA_ENV"

$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.55 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs1024-mgn001-upsamplingeven"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.65 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs1024-mgn001-upsamplingeven"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.70 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs1024-mgn001-upsamplingeven"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs1024-mgn001-upsamplingeven"
