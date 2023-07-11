"""
The results of each experiment are in a directory that looks like the following:
experiment_dir
    run_1
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...
    run_2
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...
    ...
    run_n
        seed_1
            test_results.csv
        seed_2
            test_results.csv
        ...

Each results.csv file contains the results of a single run of the experiment.
"""
import os
import re
from typing import List

import numpy as np
import pandas as pd


def gather_data_seeds(experiment_dir: str, attr_key: str, metric_names: List[str], dp: bool = False):
    """Gather the data of multiple random seeds
    For every metric, it returns a matrix of shape (num_runs, num_seeds)
    """
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir)]
    run_dirs = [run_dir for run_dir in run_dirs if run_dir.endswith('DP') and dp or not run_dir.endswith('DP') and not dp]
    run_dfs = []
    if len(run_dirs) == 0:
        return None, None
    attr_key_values = []
    for run_dir in run_dirs:
        seed_dirs = [os.path.join(run_dir, seed_dir) for seed_dir in os.listdir(run_dir)]
        seed_dfs = []
        for seed_dir in seed_dirs:
            results_file = os.path.join(seed_dir, 'test_results.csv')
            df = pd.read_csv(results_file)
            seed_dfs.append(df)
        df = pd.concat(seed_dfs)
        run_dfs.append(df)
        if "_map" in df.columns:
            # accidentally logged the map as a string
            # get value from _map column
            config_str = df["_map"].values[0]
            pattern = r"'protected_attr_percent', ([\d.]+)"
            result = re.search(pattern, config_str)
            extracted_value = float(result.group(1))
            attr_key_values.append(extracted_value)
        else:
            attr_key_values.append(df["protected_attr_percent"].values[0])
    # Sort by protected attribute
    run_dfs = [df for _, df in sorted(zip(attr_key_values, run_dfs))]
    attr_key_values = np.sort(np.array(attr_key_values))
    # Build results dictionary
    results = {metric: [] for metric in metric_names}
    for df in run_dfs:
        for metric in metric_names:
            results[metric].append(df[metric].values)
    results = {metric: np.stack(vals, axis=0) for metric, vals in results.items()}
    return results, attr_key_values


def get_lowest_seed(directory: str):
    """Get the subdirectory with the lowest seed number"""
    seed_dirs = [os.path.join(directory, seed_dir) for seed_dir in os.listdir(directory)]
    seed_dirs = [seed_dir for seed_dir in seed_dirs if os.path.isdir(seed_dir)]
    seed_nums = [int(seed_dir.split('_')[-1]) for seed_dir in seed_dirs]
    return seed_dirs[np.argmin(seed_nums)]


def avg_numeric_in_df(df: pd.DataFrame):
    """Average all columns that have numeric values"""
    def is_numeric(col):
        return np.issubdtype(col.dtype, np.number)
    for col in df.columns:
        if is_numeric(df[col]):
            df[col] = df[col].mean()
    df = df.iloc[:1]
    return df
