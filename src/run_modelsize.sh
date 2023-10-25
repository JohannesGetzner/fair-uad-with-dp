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
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 1.00 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.50 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.25 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.00 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"
# ---------
# SEX
# ---------
date=$(date '+%Y-%m-%d %H:%M:%S')
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 1.00 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.75 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.50 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.25 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"

$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 300 350 400 550      --group_name_mod  "bs32-ms" --job_type_mod "ms-300-350-400-550"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 200 250 300 450      --group_name_mod  "bs32-ms" --job_type_mod "ms-200-250-300-450"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 150 200 250 350      --group_name_mod  "bs32-ms" --job_type_mod "ms-150-200-250-350"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 100 150 200 300 400  --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200-300-400"   --d "${date}"
#$ANACONDA_PYTHON   -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 100 150 200 300      --group_name_mod  "bs32-ms" --job_type_mod "ms-default"   --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 100 150 200          --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150-200"       --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 100 150              --group_name_mod  "bs32-ms" --job_type_mod "ms-100-150"           --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 100                  --group_name_mod  "bs32-ms" --job_type_mod "ms-100"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 50                   --group_name_mod  "bs32-ms" --job_type_mod "ms-50"               --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 10                   --group_name_mod  "bs32-ms" --job_type_mod "ms-10"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 10 20                --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20"                --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 10 20 30             --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30"             --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 10 20 30 40          --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40"          --d "${date}"
$ANACONDA_PYTHON    -u train.py --run_config normal --run_version v2 --protected_attr_percent 0.00 --hidden_dims 10 20 30 40 50       --group_name_mod  "bs32-ms" --job_type_mod "ms-10-20-30-40-50"       --d "${date}"
