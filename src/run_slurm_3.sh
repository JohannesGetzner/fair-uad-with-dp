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

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --effective_dataset_size 0.002 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0002"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --effective_dataset_size 0.002 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0002"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --effective_dataset_size 0.002 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0002"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --effective_dataset_size 0.002 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0002"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --effective_dataset_size 0.005 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0005"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --effective_dataset_size 0.008 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0008"
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.10  --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_01"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.01  --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_001"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --effective_dataset_size 0.001 --d "${date}" --group_name_mod "bs32-dss" --job_type_mod "_0001"
