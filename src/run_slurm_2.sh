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
# default --hidden_dims 100 150 200 300
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200 250 300  --group_name_mod "bs32-ms" --job_type_mod "ms100-150-200-250-300"  --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 175 250 325 400  --group_name_mod "bs32-ms" --job_type_mod "ms100-175-250-325-400"  --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 200 300 400 500  --group_name_mod "bs32-ms" --job_type_mod "ms100-200-300-400-500" --d "${date}"

$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200 250 300  --group_name_mod "bs32-ms" --job_type_mod "ms100-150-200-250-300"  --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 175 250 325 400  --group_name_mod "bs32-ms" --job_type_mod "ms100-175-250-325-400"  --d "${date}"
$ANACONDA_PYTHON -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 200 300 400 500  --group_name_mod "bs32-ms" --job_type_mod "ms100-200-300-400-500" --d "${date}"
