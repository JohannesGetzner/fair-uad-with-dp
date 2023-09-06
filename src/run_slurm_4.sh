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

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 25 75 125 175    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms25-75-125-175"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 50 100 150 200   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms50-100-150-200"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 75 125 175 225   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms75-125-175-225"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200 250  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms100-150-200-250"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200 300  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "msdefault"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 125 175 225 325  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms125-175-225-325"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 150 200 250 350  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms150-200-250-350"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 175 225 275 375  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms175-225-275-375"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 200 250 300 400  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms200-250-300-400"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 25 75 125 175    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms25-75-125-175"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 50 100 150 200   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms50-100-150-200"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 75 125 175 225   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms75-125-175-225"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150 200 250  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms100-150-200-250"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150 200 300  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "msdefault"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 125 175 225 325  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms125-175-225-325"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 150 200 250 350  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms150-200-250-350"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 175 225 275 375  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms175-225-275-375"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 200 250 300 400  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms200-250-300-400"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 25 75 125 175    --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms25-75-125-175"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 50 100 150 200   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms50-100-150-200"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 75 125 175 225   --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms75-125-175-225"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200 250  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms100-150-200-250"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200 300  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "msdefault"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 125 175 225 325  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms125-175-225-325"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 150 200 250 350  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms150-200-250-350"
#$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 175 225 275 375  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms175-225-275-375"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 200 250 300 400  --d "${date}" --group_name_mod "bs32-ms" --job_type_mod "ms200-250-300-400"