"""This script is used to plot the results of the experiments."""
import sys

sys.path.append('..')
import math
import os
import re
import json
from typing import Dict, List, Tuple
import seaborn as sns
# set seaborn font-scale
sns.set(font_scale=1.2, style="whitegrid")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from einops import repeat
from scipy import stats

from utils import gather_data_seeds, combine_non_and_dp_data

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.family"] = "Times New Roman"


def plot_metric(experiment_dir: str, attr_key: str, metrics: Tuple[str], xlabel: str, ylabel: str, title: str,
                plt_name: str, dp=False):
    """Plots the given metrics as different plots"""
    # Collect data from all runs
    data, attr_key_values = gather_data_seeds(experiment_dir, attr_key, metrics, dp=dp)
    plotting_args = {
        "data": data,
        "attr_key_values": attr_key_values,
        "metrics": metrics,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "title": title,
        "experiment_dir": experiment_dir
    }
    # Plot scatter plot
    # for f, plot_name in [(plot_metric_scatter, "scatter"), (plot_metric_bar, "bar"), (plot_metric_box_whisker, "box")]:
    for f, plot_name in [(plot_metric_bar, "bar")]:
        plotting_args["plt_name"] = plt_name + '_' + plot_name + ("_DP" if dp else "") + '.png'
        f(**plotting_args)


def plot_metric_scatter(data: Dict[str, np.ndarray], attr_key_values: np.ndarray, metrics: List[str], xlabel: str,
                        ylabel: str, title: str, plt_name: str, experiment_dir: str):
    """
    Plots the given metrics as a scatter plot. Each metric is plotted in a
    separate subplot. The positions on the x-axis are slightly perturbed.
    """
    # Plot preparation
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    f = plt.figure(figsize=(6.4 * len(metrics), 4.8))
    f.suptitle(title, wrap=True)

    for i, metric in enumerate(metrics):
        ax = f.add_subplot(1, len(metrics), i + 1)
        ys = data[metric]
        # Repeat xs
        xs = repeat(centers, 'i -> i j', j=ys.shape[1])
        # Perturb xs
        xs = xs + np.random.uniform(-width / 2, width / 2, size=xs.shape)
        # Plot with color gradient
        for j, (xs_, ys_) in enumerate(zip(xs, ys)):
            c = mpl.cm.viridis(j / len(xs))
            ax.scatter(xs_, ys_, alpha=0.5, color=c)

        # Plot regression lines
        left, right = ax.get_xlim()
        y_mean = ys.mean(axis=1)
        reg_coefs = np.polyfit(centers, y_mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        ax.plot(reg_ind, reg_vals, color='black')

        # Plot standard deviation of regression lines
        all_reg_coefs = []
        for seed in range(data[metric].shape[1]):
            reg_coefs = np.polyfit(centers, ys[:, seed], 1)
            all_reg_coefs.append(reg_coefs)
        all_reg_coefs = np.stack(all_reg_coefs, axis=0)
        reg_coefs_std = all_reg_coefs.std(axis=0)
        reg_coefs_mean = all_reg_coefs.mean(axis=0)
        lower_reg_vals = np.polyval(reg_coefs_mean - reg_coefs_std, reg_ind)
        upper_reg_vals = np.polyval(reg_coefs_mean + reg_coefs_std, reg_ind)
        plt.fill_between(reg_ind, lower_reg_vals, upper_reg_vals, color='black', alpha=0.2)
        # Significance test of regression line
        _, _, _, p_value, _ = stats.linregress(centers, y_mean)
        alpha = 0.05
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}")

        # Plot settings
        ax.set_title(metric)

        # x label and x ticks
        ax.set_xlabel(xlabel)
        ax.set_xticks(centers, attr_key_values.round(2))
        ax.set_xlim(left, right)

        # y label and y ticks
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_yticklabels([])

    # All axes should have the same y limits
    ylim_min = min([ax.get_ylim()[0] for ax in f.axes])
    ylim_max = max([ax.get_ylim()[1] for ax in f.axes])
    for ax in f.axes:
        ax.set_ylim(ylim_min, ylim_max)

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(experiment_dir, plt_name))
    plt.close()


def plot_metric_bar(data: Dict[str, np.ndarray], attr_key_values: np.ndarray, metrics: List[str], xlabel: str,
                    ylabel: str, title: str, plt_name: str, experiment_dir: str):
    # Prepare plot
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    bars = []
    mini = math.inf
    maxi = -math.inf

    # Plot bar plots
    scores = {}
    for i, metric in enumerate(metrics):
        mean = data[metric].mean(axis=1)
        std = data[metric].std(axis=1)
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        yerr = np.stack([mean - lower, upper - mean], axis=0)
        min_val = np.min(lower)
        max_val = np.max(upper)
        bars.append(plt.bar(ind + i * width, mean, width=width, yerr=yerr, ecolor='darkgray'))
        scores[metric] = mean.tolist()
        scores[f"{metric}_err"] = yerr.tolist()
        if mini > min_val:
            mini = min_val
        if maxi < max_val:
            maxi = max_val
    # save scores as json
    with open(os.path.join(experiment_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f)

    # Plot regression lines
    left, right = plt.xlim()
    for i, metric in enumerate(metrics):
        vals = data[metric]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(centers, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color = list(bars[i][0].get_facecolor())  # get color of corresponding bar
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)
        # Significance test of regression line
        _, _, _, p_value, _ = stats.linregress(centers, mean)
        alpha = 0.05
        is_significant = p_value < alpha
        print(f"The regression line of {metric} is significantly different from 0: {is_significant}")

    # Plot settings
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2))
    plt.xlim(left, right)
    legend_labels = [re.search(r"(old|female|male|young)", m).group(1) for m in metrics]
    plt.legend(bars,legend_labels)
    ylim_min = max(0, mini - 0.1 * (maxi - mini))
    plt.ylim(ylim_min, plt.ylim()[1])
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(experiment_dir, plt_name))
    plt.close()


def plot_metric_bar_compare(df, groups, fig_size=(10, 6)):
    plt.figure(figsize=fig_size)
    colors = ["#336699", "#6688AA", "#FF9933", "#FFB366"]
    # create bar-plot
    df.sort_values(by="group", key=lambda column: column.map(lambda e: groups.index(e)), inplace=True)
    g = sns.barplot(x="percent", y="value", hue="group", data=df, palette=colors, errorbar="ci")
    g.set(ylim=(0.5, 1.0))
    # plot regression lines
    for idx, group in enumerate(df["group"].unique()):
        group_data = df[df["group"] == group]
        group_data["percent"] = group_data["percent"] * 4
        # TODO: regression and statistical significance
        slope, intercept, r_value, p_value, std_err = linregress(group_data["percent"], group_data["value"])
        if p_value < 0.05:
            g = sns.regplot(
                x="percent",
                y="value",
                data=group_data,
                scatter=False,
                line_kws={"label": f"{group} regression line", "color": colors[idx], "lw": 1.7, "ls": "--"},
                truncate=False
            )
    # set axis labels
    plt.xlabel(f"percent {groups[0]} samples in training data")
    plt.ylabel("subgroup AUROC")
    return g

def plot_metric_box_whisker(data: Dict[str, np.ndarray], attr_key_values: np.ndarray, metrics: List[str], xlabel: str,
                            ylabel: str, title: str, plt_name: str, experiment_dir: str):
    """
    Plots the given metrics as a box and whisker plot.
    """
    # Prepare plot
    width = 0.25
    ind = np.arange(len(data[metrics[0]]))
    centers = ind + (len(metrics) // 2) * width - width / 2
    boxes = []

    # Plot bar plots
    for i, metric in enumerate(metrics):
        ys = data[metric].T
        # Repeat xs
        positions = ind + i * width
        # Plot with diamond markers
        bplot = plt.boxplot(ys, positions=positions, widths=width, showfliers=False, boxprops={'color': 'gray'},
                            whiskerprops={'color': 'gray'}, capprops={'color': 'gray'},
                            medianprops={'color': 'darkgray'}, patch_artist=True)
        boxes.append(bplot)

        # Set colors
        color = 'tab:blue' if i == 0 else 'tab:orange'
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    # Plot regression lines
    left, right = plt.xlim()
    for i, metric in enumerate(metrics):
        vals = data[metric]
        mean = vals.mean(axis=1)
        reg_coefs = np.polyfit(centers, mean, 1)  # linear regression
        reg_ind = np.array([left, right])
        reg_vals = np.polyval(reg_coefs, reg_ind)
        color = list(boxes[i]['boxes'][0].get_facecolor())  # get color of corresponding box
        color[-1] = 0.5  # set alpha to 0.5
        plt.plot(reg_ind, reg_vals, color=color)

    # Plot settings
    plt.legend([bplot["boxes"][0] for bplot in boxes], metrics)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, wrap=True)
    plt.xticks(centers, attr_key_values.round(2))

    # Save plot
    print(f"Saving plot to {plt_name}")
    plt.savefig(os.path.join(experiment_dir, plt_name))
    plt.close()


if __name__ == '__main__':
    to_skip = ["2023-08-07 16:45:30-FAE-rsna-age-bs1024_mgn001_old_down_weighted-DP","2023-08-13 09:52:07-FAE-rsna-age-bs32_old_down_weighted-noDP"]
    dir = ""
    if dir == "":
        dirs = os.listdir("../logs_persist")
    else:
        dirs = [dir]
    for experiment_dir in dirs:
        if experiment_dir in to_skip:
            continue
        print("\nGenerating plots for", experiment_dir, "\n")
        experiment_dir = os.path.join("../logs_persist", experiment_dir)
        if "age" in experiment_dir:
            pv = "age"
            g = ("old", "young")
        else:
            pv = "sex"
            g = ("male", "female")
        metrics = [
            # fpr@0.95
            # (f"test/lungOpacity_{g[0]}_fpr@0.95", f"test/lungOpacity_{g[1]}_fpr@0.95", "fpr@0.95tpr"),
            # tpr@0.05
            # (f"test/lungOpacity_{g[0]}_tpr@0.05", f"test/lungOpacity_{g[1]}_tpr@0.05", "tpr@0.05fpr"),
            # anomaly score
            # (f"test/lungOpacity_{g[0]}_anomaly_score", f"test/lungOpacity_{g[1]}_anomaly_score", "anomaly score"),
            # AUROC
            # (f"test/lungOpacity_{g[0]}_AUROC", f"test/lungOpacity_{g[1]}_AUROC", "AUROC"),
            # subgroupAUROC
            (f"test/lungOpacity_{g[0]}_subgroupAUROC", f"test/lungOpacity_{g[1]}_subgroupAUROC", "subgroupAUROC"),
            # Average precision
            # (f"test/lungOpacity_{g[0]}_AP", f"test/lungOpacity_{g[1]}_AP", "AP")
        ]

        for metric in metrics:
            title = f"FAE {metric[2]} on RSNA for different proportions of {g[0]} patients in training"
            plot_metric(
                experiment_dir=experiment_dir,
                metrics=metric[:2],
                attr_key=f'{g[0]}_percent',
                xlabel=f"percentage of {g[0]} subjects in training set",
                ylabel=metric[2],
                title=title,
                plt_name=f"fae_rsna_{pv}_{metric[2]}",
                dp="noDP" not in experiment_dir
            )

    if False:
        non_dp_dir = os.path.join("../logs_persist", "2023.07.11-20:38:32-FAE-rsna-age-bs32-noDP")
        dp_dir = os.path.join("../logs_persist", "2023.07.13-09:20:21-FAE-rsna-age-bs1024_mgn001-DP")
        metrics = (f"test/lungOpacity_old_subgroupAUROC", f"test/lungOpacity_young_subgroupAUROC")
        groups = ["old", "young", "old_dp", "young_dp"]
        df = combine_non_and_dp_data(
            non_dp_dir,
            dp_dir,
            "old_percent",
            metrics,
            ["old", "young", "old_dp", "young_dp"]
        )
        g = plot_metric_bar_compare(df, groups)
        plt.show()