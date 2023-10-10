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
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.6  --upsampling_strategy "even_young"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.7  --upsampling_strategy "even_young"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --upsampling_strategy "even_young"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.8  --upsampling_strategy "even_young"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.9  --upsampling_strategy "even_young"	--group_name_mod "bs32-upsamplingeven" --d "${date}"

# ---------
# SEX
# ---------
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.6  --upsampling_strategy "even_female"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.7  --upsampling_strategy "even_female"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --upsampling_strategy "even_female"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.8  --upsampling_strategy "even_female"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.9  --upsampling_strategy "even_female"	--group_name_mod "bs32-upsamplingeven" --d "${date}"
