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

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.01 --upsampling_strategy "even-male" --d "${date}" --group_name_mod "bs32-upsamplingevenmale"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.10 --upsampling_strategy "even-male" --d "${date}" --group_name_mod "bs32-upsamplingevenmale"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.20 --upsampling_strategy "even-male" --d "${date}" --group_name_mod "bs32-upsamplingevenmale"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.30 --upsampling_strategy "even-male" --d "${date}" --group_name_mod "bs32-upsamplingevenmale"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.40 --upsampling_strategy "even-male" --d "${date}" --group_name_mod "bs32-upsamplingevenmale"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.001 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.01 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.10 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.20 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.30 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.40 --upsampling_strategy "even-old" --d "${date}" --group_name_mod "bs32-upsamplingevenold"