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
from typing import List, Tuple

import numpy as np
import pandas as pd


def gather_data_seeds(experiment_dir: str, attr_key: str, metric_names):
    """Gather the data of multiple random seeds
    For every metric, it returns a matrix of shape (num_runs, num_seeds)
    """
    test = os.listdir(experiment_dir)
    run_dirs = [os.path.join(experiment_dir, run_dir) for run_dir in os.listdir(experiment_dir) if not run_dir.endswith('.png') and not run_dir.endswith('.json') and not run_dir.endswith('.ipynb')]
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
            # print("PLEASE EVALUATE")
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
    # check that all arrays in results have the same length
    max_length = max(len(arr) for arr in results[metric])
    for metric in metric_names:
        results[metric] = [np.concatenate((arr, [arr[-1]] * (max_length - len(arr)))) for arr in results[metric]]
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


def compare_two_runs(exp_dir: str, exp_dir_two: str, attr_key: str, metrics: Tuple[str], group_names: List[str]):
    data, attr_key_values = gather_data_seeds(exp_dir, attr_key, metrics)
    data_two, attr_key_values_two = gather_data_seeds(exp_dir_two, attr_key, metrics)

    df = pd.DataFrame(columns=["percent", "value", "group"])
    for metric_name, metric_values in data.items():
        tmp = pd.DataFrame(metric_values)
        num_cols = tmp.shape[1]
        for idx, row in tmp.iterrows():
            for i in range(num_cols):
                bar_label = group_names[0] if group_names[0] in metric_name else group_names[1]
                new_row = {"percent": attr_key_values[idx], "group": bar_label, "value": row[i]}
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])])

    for metric_name, metric_values in data_two.items():
        tmp = pd.DataFrame(metric_values)
        num_cols = tmp.shape[1]
        for idx, row in tmp.iterrows():
            for i in range(num_cols):
                bar_label = group_names[2] if group_names[0] in metric_name else group_names[3]
                new_row = {"percent": attr_key_values_two[idx], "group": bar_label, "value": row[i]}
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])])
    return df