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

# --------------
# AGE DP
# --------------
date=$(date '+%Y-%m-%d %H:%M:%S')
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --loss_weight_type "old_weight" --weight 0.9	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --loss_weight_type "old_weight" --weight 0.5	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --loss_weight_type "old_weight" --weight 0.1	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --loss_weight_type "old_weight" --weight 0.001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.25 --loss_weight_type "old_weight" --weight 0.0001	--no_img_log   --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"

#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --loss_weight_type "old_weight" --weight 0.9	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --loss_weight_type "old_weight" --weight 0.5	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --loss_weight_type "old_weight" --weight 0.1	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --loss_weight_type "old_weight" --weight 0.001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.50 --loss_weight_type "old_weight" --weight 0.0001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"

#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --loss_weight_type "old_weight" --weight 0.9	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --loss_weight_type "old_weight" --weight 0.5	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --loss_weight_type "old_weight" --weight 0.1	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --loss_weight_type "old_weight" --weight 0.001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v1 --protected_attr_percent 0.75 --loss_weight_type "old_weight" --weight 0.0001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"

# --------------
# SEX DP
# --------------
date=$(date '+%Y-%m-%d %H:%M:%S')
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --loss_weight_type "male_weight" --weight 0.9	   --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --loss_weight_type "male_weight" --weight 0.5	   --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --loss_weight_type "male_weight" --weight 0.1	   --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --loss_weight_type "male_weight" --weight 0.001	 --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.25 --loss_weight_type "male_weight" --weight 0.0001	 --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"

$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --loss_weight_type "male_weight" --weight 0.9	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --loss_weight_type "male_weight" --weight 0.5	  --no_img_log   --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --loss_weight_type "male_weight" --weight 0.1	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --loss_weight_type "male_weight" --weight 0.001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.50 --loss_weight_type "male_weight" --weight 0.0001	--no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"

$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --loss_weight_type "male_weight" --weight 0.9	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-09" --d "${date}"
#$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --loss_weight_type "male_weight" --weight 0.5	  --no_img_log   --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-05" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --loss_weight_type "male_weight" --weight 0.1	    --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-01" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --loss_weight_type "male_weight" --weight 0.001	  --no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-001" --d "${date}"
$ANACONDA_PYTHON -u   train.py --run_config dp --run_version v2 --protected_attr_percent 0.75 --loss_weight_type "male_weight" --weight 0.0001	--no_img_log --group_name_mod "bs512-lossweight"  --job_type_mod "oldweight-0001" --d "${date}"