#!/bin/bash
date=$(date '+%Y-%m-%d %H:%M:%S')
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.55 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.65 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.70 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.80 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.85 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.90 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.95 --upsampling_strategy "even" --d "${date}" --group_name_mod "bs32-upsamplingeven"
date=$(date '+%Y-%m-%d %H:%M:%S')
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.55 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.65 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.70 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.75 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.80 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.85 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.90 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"
python -u train.py --run_config normal --run_version v1 --protected_attr_percent 0.95 --upsampling_strategy "random" --d "${date}" --group_name_mod "bs32-upsamplingrandom"