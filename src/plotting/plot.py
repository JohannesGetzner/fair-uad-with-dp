import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
# set whitegrid but don't modify rcParams
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
from typing import List
from scipy.stats import linregress
from matplotlib.patches import Rectangle

#from matplotlib.font_manager import fontManager, FontProperties
#path = "/home/getznerj/Downloads/Palatino Font Free/Palatino.ttf"
#fontManager.addfont(path)
#prop = FontProperties(fname=path)
#plt.rcParams['font.family'] = prop.get_name()

PV_MAP = {
    "age": ("old","young"),
    "sex": ("male", "female")
}

TUM_colors =  {
    "TUMBlue": "#0065BD",
    "TUMSecondaryBlue": "#005293",
    "TUMSecondaryBlue2": "#003359",
    "TUMBlack": "#000000",
    "TUMWhite": "#FFFFFF",
    "TUMDarkGray": "#333333",
    "TUMGray": "#808080",
    "TUMLightGray": "#CCCCC6",
    "TUMAccentGray": "#DAD7CB",
    "TUMAccentOrange": "#E37222",
    "TUMAccentGreen": "#A2AD00",
    "TUMAccentLightBlue": "#98C6EA",
    "TUMAccentBlue": "#64A0C8"
}

def load_logs(log_dir = "", fine_tuning= False):
    run_dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    # each log dir contains a set of runs at a given split
    results = {}
    for run_dir in run_dirs:
        results[run_dir] = []
        run_split_dirs = [os.path.join(run_dir, d) for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))]
        # each run split dir contains a set of runs (different seeds)
        for run_split_dir in run_split_dirs:
            seed_dirs = [os.path.join(run_split_dir, d) for d in os.listdir(run_split_dir) if os.path.isdir(os.path.join(run_split_dir, d))]
            if fine_tuning:
                dp = not "noDP" in run_dir
                protected_attr_percent = run_split_dir.split("_")[-1 if not dp else -2]
            else:
                protected_attr_percent = run_split_dir.split("_")[-1]
            if len(protected_attr_percent) > 1:
                # add a '.' after the first 0
                protected_attr_percent = protected_attr_percent[:1] + "." + protected_attr_percent[1:]
            for seed_dir in seed_dirs:
                # each seed contains a csv with the results of the run
                try:
                    results_seed = pd.read_csv(os.path.join(seed_dir, "test_results.csv" if not fine_tuning else "test_results_stage_two.csv"))
                except:
                    results_seed = pd.read_csv(os.path.join(seed_dir, "test_results_.csv"))
                if fine_tuning:
                    results_seed["protected_attr_percent"] = float(protected_attr_percent)
                results[run_dir].append(results_seed)
        # concat all seeds and replace
        results[run_dir] = pd.concat(results[run_dir])
        # reset index
        results[run_dir].reset_index(inplace=True, drop=True)
    return results


def plot_runs(df,
              metrics,
              pv,
              secondary_color_at: float = None,
              secondary_color_legend_text: str = "",
              x_var="protected_attr_percent",
              regress=True,
              override_insignificance=False,
              ax=None,
              font_size = 11,
              legend_outside=False,
              ylim = 50
              ):
    palette = {
        "old": TUM_colors["TUMAccentBlue"], "young":TUM_colors["TUMAccentOrange"],
        "male": TUM_colors["TUMAccentBlue"], "female": TUM_colors["TUMAccentOrange"]
    }
    if pv == "age":
        metrics = metrics[::-1]
    id_vars = ["protected_attr", "protected_attr_percent", "seed"]
    if x_var != "protected_attr_percent":
        id_vars.append(x_var)

    df_melted = df.melt(id_vars=id_vars, value_vars=metrics,
                        var_name=f"subgroup by {pv}", value_name="value")
    df_melted[f"subgroup by {pv}"] = pd.Categorical(df_melted[f"subgroup by {pv}"], categories=metrics)
    df_melted["value"] = df_melted["value"] * 100
    df_melted["value"] = df_melted["value"].astype(int)
    # multiply protected_attr_percent by 100
    df_melted["protected_attr_percent"] = (df_melted["protected_attr_percent"] * 100).astype(int)
    g = sns.barplot(x=x_var, y="value", hue=f"subgroup by {pv}", data=df_melted, errorbar="ci", ax=ax, palette=palette)

    percent_values = sorted(df_melted[x_var].unique())
    if secondary_color_at:
        secondary_color_at_idx = percent_values.index(secondary_color_at)
        for i in [0, len(percent_values)]:
            bars = g.patches
            bars[secondary_color_at_idx + i].set(linestyle="--", edgecolor="red", linewidth=1.5)

    bar_means = [patch.get_height() for patch in g.patches]
    cis = [line.get_xydata() for line in g.lines]
    for idx, patch in enumerate(g.patches):
        #print(f"bar at {patch.get_x() + patch.get_width() / 2} has mean {bar_means[idx]:.3f} +- {cis[idx][1][1] - bar_means[idx]:.3f}")
        if idx < len(percent_values):
            if bar_means[idx] > bar_means[idx + len(percent_values)]:
                y = cis[idx][1][1] + 0.065*(100-ylim)
            else:
                y = cis[idx][0][1]  - 0.02*(100-ylim)
        else:
            if bar_means[idx] > bar_means[idx - len(percent_values)]:
                y = cis[idx][1][1] + 0.065*(100-ylim)
            else:
                y = cis[idx][0][1]  - 0.02*(100-ylim)
        ax.text(x=patch.get_x() + patch.get_width() / 2, y=y, s=f"{bar_means[idx]:.0f}", ha='center', va='top',
            color='black', fontsize=font_size*0.8)
    x_coords = sorted([patch.get_x() + patch.get_width() / 2 for patch in g.patches])
    regression_lines = []
    p_value = np.inf
    if regress:
        for idx, metric in enumerate(metrics):
            df_melted_sub = df_melted[df_melted[f"subgroup by {pv}"] == metric]
            # choose every second value starting from idx (differentiate bars corresponding to different metrics)
            x_coords_sub = x_coords[idx::2]
            percent_values = sorted(df_melted_sub["protected_attr_percent"].unique())
            percent_to_x_coord = {p: x for p, x in zip(percent_values, x_coords_sub)}
            df_melted_sub.loc[:, "x_coord"] = df_melted_sub.loc[:, "protected_attr_percent"].map(percent_to_x_coord)
            slope, intercept, r_value, p_value, std_err = linregress(df_melted_sub["protected_attr_percent"],
                                                                     df_melted_sub["value"])
            print(f"Slope for {metric} is {slope:.3f} with p-value {p_value:.3f} ")
            if p_value < 0.05 or override_insignificance:
                line_label = f"{metric} regressed"
                color = palette[metric]
                regression_lines.append(plt.Line2D([0], [0], color=color, linestyle='--', label=line_label))
                sns.regplot(x="x_coord", y="value", data=df_melted_sub, scatter=False,
                    line_kws={"color": color, "lw": 1.7, "ls": "--"}, truncate=False, ci=95, ax=ax)

    handles, labels = ax.get_legend_handles_labels()

    #if p_value < 0.05 or override_insignificance:
    #    handles += regression_lines
    #    labels += [line.get_label() for line in regression_lines]

    # add a Rectangle to the legend not Line2D
    if secondary_color_at:
        handles.append(Rectangle((0, 0), 1, 1, fc='none', ec='red', linestyle='--', label='Legend Box'))
        labels.append(secondary_color_legend_text)

    if legend_outside:
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        ax.legend(handles, labels, loc='upper left')
    ax.set(ylim=(ylim, 100))
    if x_var == "weight":
        ax.set_xlabel(f"Weight applied to {'Old' if pv == 'age' else 'Male'} Loss")
        x_ticks_long = ["1e-4", "5e-4", "1e-3", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        x_ticks_short = ["1e-4", "1e-3", "0.1", "0.9", "1.0"]
        g.set_xticks(range(len(x_ticks_short if len(ax.get_xticks()) <= 5 else x_ticks_long)))
        g.set_xticklabels(x_ticks_short if len(ax.get_xticks()) <= 5 else x_ticks_long)
    else:
        ax.set_xlabel(f"Percentage of {'Old' if pv == 'age' else 'Male'} Samples in Training Set")
    ax.set_ylabel("s-AUC")
    return ax

def add_values_from_normal_runs(log_dir, data_outer, split):
    data_normal = load_logs(log_dir = os.path.join(log_dir, "baseline"))
    for key, df_normal in data_normal.items():
        pv = "age" if "age" in key else "sex"
        dp = False if "noDP" in key else True
        corresponding_outer_key = [k for k in data_outer.keys() if pv in k and ((not dp) == ("noDP" in k))][0]
        rows = df_normal[df_normal["protected_attr_percent"] == split]
        data_outer[corresponding_outer_key] = pd.concat([data_outer[corresponding_outer_key],rows], axis=0)
    return data_outer