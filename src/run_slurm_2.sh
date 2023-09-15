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


$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 3 6 9 12    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-6-9-12"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 3 8 13 18   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms3-8-13-18"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 5 10 15 20  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms5-10-15-20"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 15 20 25 30 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms15-20-25-30"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 35 40 45 50 --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms35-40-45-50"