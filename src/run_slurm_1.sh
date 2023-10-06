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

$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 0.001 --group_name_mod "bs512-dss" --job_type_mod "0001" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 0.005 --group_name_mod "bs512-dss" --job_type_mod "0005" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 0.010 --group_name_mod "bs512-dss" --job_type_mod "001" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 0.100 --group_name_mod "bs512-dss" --job_type_mod "01" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 0.700 --group_name_mod "bs512-dss" --job_type_mod "07" --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v2 --protected_attr_percent 0.5 --effective_dataset_size 1.000 --group_name_mod "bs512-dss" --job_type_mod "1" --d "${date}"

date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON -u train.py --run_config dp --run_version v1 --protected_attr_percent 0.5 --effective_dataset_size 1.0  --group_name_mod "bs512-dss" --job_type_mod "1" --d "${date}"