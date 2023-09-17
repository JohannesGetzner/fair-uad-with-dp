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
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 1 1 1 1  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-1-1-1"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 1 1 2 2  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-1-2-2"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 1 1 1 1  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-1-1-1"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 1 1 2 2  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-1-2-2"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 1 2 3 4  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-2-3-4"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 1 3 5 7  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-3-5-7"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 1 4 7 10 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-4-7-10"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.5 --hidden_dims 2 5 8 11 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms2-5-8-11"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 1 2 3 4  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-2-3-4"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 1 3 5 7  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-3-5-7"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 1 4 7 10 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms1-4-7-10"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 2 5 8 11 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms2-5-8-11"