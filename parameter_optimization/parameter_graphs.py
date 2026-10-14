import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

FILE = "parameter_results.csv"
OUT = "ParameterGraphs"

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(FILE)

# remove repeated headers
df = df[df["rep"].astype(str) != "rep"].copy()

# convert numeric columns
numeric = ["rep", "noise", "layers", "lr", "eval_acc"]

for column in numeric:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# only use reps 0-7
df = df[df["rep"].between(0, 7)]


def make_graph(column, title, xlabel, filename, categorical=False):

    # first calculate one mean for each rep
    rep_means = (
        df.groupby(["rep", column])["eval_acc"]
        .mean()
        .reset_index()
    )

    # then calculate statistics across reps
    stats = (
        rep_means.groupby(column)["eval_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats["se"] = stats["std"] / np.sqrt(stats["count"])
    stats["ci95"] = (
        t.ppf(0.975, stats["count"] - 1)
        * stats["se"]
    )

    if not categorical:
        stats = stats.sort_values(column)

    plt.figure(figsize=(7, 5))

    if categorical:
        plt.errorbar(
            stats[column],
            stats["mean"],
            yerr=stats["ci95"],
            fmt="o",
            capsize=5
        )
    else:
        plt.errorbar(
            stats[column],
            stats["mean"],
            yerr=stats["ci95"],
            marker="o",
            capsize=5
        )

    plt.xlabel(xlabel)
    plt.ylabel("Mean evaluation accuracy")
    plt.title(title)
    plt.grid(alpha=.3)

    if not categorical:
        plt.xticks(stats[column])

    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}", dpi=200)
    plt.close()

    print("\n" + title)
    print(stats[[column, "mean", "ci95", "count"]])


# 1. variational layers
make_graph(
    "layers",
    "Accuracy vs Variational Layers",
    "Variational layers",
    "1_layers.png"
)

# 2. learning rate
make_graph(
    "lr",
    "Accuracy vs Learning Rate",
    "Learning rate",
    "2_learning_rate.png"
)

# 3. feature map
make_graph(
    "feature_map",
    "Accuracy by Feature Map",
    "Feature map",
    "3_feature_map.png",
    categorical=True
)

# 4. dataset
make_graph(
    "dataset",
    "Accuracy by Dataset",
    "Dataset",
    "4_dataset.png",
    categorical=True
)

# 5. noise level
make_graph(
    "noise",
    "Accuracy vs Noise Level",
    "Noise level",
    "5_noise.png"
)

print("\nCreated 5 graphs in", OUT)