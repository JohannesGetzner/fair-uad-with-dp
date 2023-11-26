import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import linregress
from matplotlib.patches import Rectangle

PV_MAP = {
    "age": ("old","young"),
    "sex": ("male", "female")
}

def load_logs(log_dir = ""):
    run_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    # each log dir contains a set of runs at a given split
    results = {}
    for run_dir in run_dirs:
        results[run_dir] = []
        run_split_dirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        # each run split dir contains a set of runs (different seeds)
        for run_split_dir in run_split_dirs:
            seed_dirs = [os.path.join(run_split_dir, d) for d in os.listdir(run_split_dir) if os.path.isdir(os.path.join(run_split_dir, d))]
            for seed_dir in seed_dirs:
                # each seed contains a csv with the results of the run
                results_seed = pd.read_csv(os.path.join(seed_dir, "test_results.csv"))
                results[run_dir].append(results_seed)
        # concat all seeds and replace
        results[run_dir] = pd.concat(results[run_dir])
        # reset index
        results[run_dir].reset_index(inplace=True, drop=True)
    return results


def plot_runs(df, metrics, pv, figsize=(12, 9), secondary_color_at: int = None, secondary_color_legend_text: str = ""):
    sns.set(font_scale=1.2, style="whitegrid")
    df_melted = df.melt(id_vars=["protected_attr", "protected_attr_percent", "seed"], value_vars=metrics,
                        var_name=f"subgroup by {pv}", value_name="value")
    df_melted[f"subgroup by {pv}"] = pd.Categorical(df_melted[f"subgroup by {pv}"], categories=metrics)
    df_melted.sort_values(by=[f"subgroup by {pv}"], inplace=True)
    plt.figure(figsize=figsize)
    g = sns.barplot(x="protected_attr_percent", y="value", hue=f"subgroup by {pv}", data=df_melted, errorbar="ci", )
    percent_values = sorted(df_melted["protected_attr_percent"].unique())
    if secondary_color_at:
        secondary_color_at_idx = percent_values.index(secondary_color_at)
        for i in [0, len(percent_values)]:
            bars = g.patches
            bars[secondary_color_at_idx + i].set(linestyle="--", edgecolor="red", linewidth=1.5)

    bar_means = [patch.get_height() for patch in g.patches]
    cis = [line.get_xydata() for line in g.lines]
    for idx, patch in enumerate(g.patches):
        # print bar mean below ci
        # get percent value of bar
        print(
            f"bar at {patch.get_x() + patch.get_width() / 2} has mean {bar_means[idx]:.3f} +- {cis[idx][1][1] - bar_means[idx]:.3f}")
        if idx < len(percent_values):
            if bar_means[idx] > bar_means[idx + len(percent_values)]:
                y = cis[idx][1][1] + 0.011
            else:
                y = cis[idx][0][1]  # - 0.008
        else:
            if bar_means[idx] > bar_means[idx - len(percent_values)]:
                y = cis[idx][1][1] + 0.011
            else:
                y = cis[idx][0][1]  # - 0.008
        plt.text(x=patch.get_x() + patch.get_width() / 2, y=y, s=f"{bar_means[idx]:.2f}", ha='center', va='top',
            color='black', fontsize=11)
    x_coords = sorted([patch.get_x() + patch.get_width() / 2 for patch in g.patches])
    regression_lines = []
    for idx, metric in enumerate(metrics):
        df_melted_sub = df_melted[df_melted[f"subgroup by {pv}"] == metric]
        # choose every second value starting from idx (differentiate bars corresponding to different metrics)
        x_coords_sub = x_coords[idx::2]
        percent_values = sorted(df_melted_sub["protected_attr_percent"].unique())
        percent_to_x_coord = {p: x for p, x in zip(percent_values, x_coords_sub)}
        df_melted_sub.loc[:, "x_coord"] = df_melted_sub.loc[:, "protected_attr_percent"].map(percent_to_x_coord)
        slope, intercept, r_value, p_value, std_err = linregress(df_melted_sub["protected_attr_percent"],
                                                                 df_melted_sub["value"])
        if p_value < 0.05:
            line_label = f"{metric} regressed"
            color = sns.color_palette()[idx]
            regression_lines.append(plt.Line2D([0], [0], color=color, linestyle='--', label=line_label))
            g = sns.regplot(x="x_coord", y="value", data=df_melted_sub, scatter=False,
                line_kws={"color": color, "lw": 1.7, "ls": "--"}, truncate=False, ci=95)
    handles, labels = plt.gca().get_legend_handles_labels()

    if p_value < 0.05:
        handles += regression_lines
        labels += [line.get_label() for line in regression_lines]

    # add a Rectangle to the legend not Line2D
    if secondary_color_at:
        handles.append(Rectangle((0, 0), 1, 1, fc='none', ec='red', linestyle='--', label='Legend Box'))
        labels.append(secondary_color_legend_text)

    plt.legend(handles, labels, loc='best')
    g.set(ylim=(0.5, 1.0))
    plt.yticks(np.arange(0.5, 1.0, 0.05))
    plt.xlabel("Percentage of Old Samples in Training Set")
    plt.ylabel("subgroup AUROC")
    plt.show()
    return g

def add_values_from_normal_runs(data_outer, split):
    data_normal = load_logs(log_dir = "../logs_final/normal")
    for key, df_normal in data_normal.items():
        pv = "age" if "age" in key else "sex"
        dp = False if "noDP" in key else True
        corresponding_outer_key = [k for k in data_outer.keys() if pv in k and ((not dp) == ("noDP" in k))][0]
        rows = df_normal[df_normal["protected_attr_percent"] == split]
        data_outer[corresponding_outer_key] = pd.concat([data_outer[corresponding_outer_key],rows], axis=0)
    return data_outer