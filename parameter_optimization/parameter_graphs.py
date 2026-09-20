import os
import pandas as pd
import matplotlib.pyplot as plt

FILE = "parameter_results_stats.csv"
OUT = "ParameterGraphs"

os.makedirs(OUT, exist_ok=True)
df = pd.read_csv(FILE)


def get(summary):
    return df[df["summary"] == summary].copy()


# 1. accuracy vs layers

x = get("layers").sort_values("layers")

plt.figure(figsize=(7, 5))
plt.errorbar(
    x["layers"],
    x["mean_eval_acc"],
    yerr=x["se_eval_acc"],
    marker="o",
    capsize=4
)
plt.xlabel("Variational layers")
plt.ylabel("Mean evaluation accuracy")
plt.title("Accuracy vs Circuit Depth")
plt.xticks(x["layers"])
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/1_layers.png", dpi=200)
plt.close()


# 2. accuracy vs learning rate

x = get("learning_rate").sort_values("lr")

plt.figure(figsize=(7, 5))
plt.errorbar(
    x["lr"],
    x["mean_eval_acc"],
    yerr=x["se_eval_acc"],
    marker="o",
    capsize=4
)
plt.xlabel("Learning rate")
plt.ylabel("Mean evaluation accuracy")
plt.title("Accuracy vs Learning Rate")
plt.xticks(x["lr"])
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/2_learning_rate.png", dpi=200)
plt.close()


# 3. accuracy by feature map

x = get("feature_map")

plt.figure(figsize=(7, 5))
plt.bar(
    x["feature_map"].str.upper(),
    x["mean_eval_acc"],
    yerr=x["se_eval_acc"],
    capsize=4
)
plt.xlabel("Feature map")
plt.ylabel("Mean evaluation accuracy")
plt.title("Accuracy by Feature Map")
plt.tight_layout()
plt.savefig(f"{OUT}/3_feature_maps.png", dpi=200)
plt.close()


# 4. layers x learning rate

x = get("layers_x_learning_rate")

plt.figure(figsize=(7, 5))

for layers in sorted(x["layers"].unique()):
    y = x[x["layers"] == layers].sort_values("lr")
    plt.plot(
        y["lr"],
        y["mean_eval_acc"],
        marker="o",
        label=f"{int(layers)} layers"
    )

plt.xlabel("Learning rate")
plt.ylabel("Mean evaluation accuracy")
plt.title("Learning Rate × Circuit Depth")
plt.xticks(sorted(x["lr"].unique()))
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/4_layers_learning_rate.png", dpi=200)
plt.close()


# 5. feature map x layers

x = get("feature_map_x_layers")

plt.figure(figsize=(7, 5))

for feature_map in x["feature_map"].unique():
    y = x[x["feature_map"] == feature_map].sort_values("layers")
    plt.plot(
        y["layers"],
        y["mean_eval_acc"],
        marker="o",
        label=feature_map.upper()
    )

plt.xlabel("Variational layers")
plt.ylabel("Mean evaluation accuracy")
plt.title("Feature Map Performance vs Circuit Depth")
plt.xticks(sorted(x["layers"].unique()))
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/5_feature_map_layers.png", dpi=200)
plt.close()


# 6. feature map x learning rate

x = get("feature_map_x_learning_rate")

plt.figure(figsize=(7, 5))

for feature_map in x["feature_map"].unique():
    y = x[x["feature_map"] == feature_map].sort_values("lr")
    plt.plot(
        y["lr"],
        y["mean_eval_acc"],
        marker="o",
        label=feature_map.upper()
    )

plt.xlabel("Learning rate")
plt.ylabel("Mean evaluation accuracy")
plt.title("Feature Map Performance vs Learning Rate")
plt.xticks(sorted(x["lr"].unique()))
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/6_feature_map_learning_rate.png", dpi=200)
plt.close()


# 7. feature map x noise

x = get("feature_map_x_noise")

plt.figure(figsize=(7, 5))

for feature_map in x["feature_map"].unique():
    y = x[x["feature_map"] == feature_map].sort_values("noise")
    plt.plot(
        y["noise"],
        y["mean_eval_acc"],
        marker="o",
        label=feature_map.upper()
    )

plt.xlabel("Noise level")
plt.ylabel("Mean evaluation accuracy")
plt.title("Feature Map Performance vs Noise")
plt.xticks(sorted(x["noise"].unique()))
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/7_feature_map_noise.png", dpi=200)
plt.close()


# 8. accuracy by dataset

x = get("dataset").sort_values("mean_eval_acc")

plt.figure(figsize=(8, 5))
plt.bar(
    x["dataset"],
    x["mean_eval_acc"],
    yerr=x["se_eval_acc"],
    capsize=4
)
plt.xlabel("Dataset")
plt.ylabel("Mean evaluation accuracy")
plt.title("Accuracy by Dataset")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{OUT}/8_datasets.png", dpi=200)
plt.close()


# 9. convergence steps vs learning rate

x = get("learning_rate").sort_values("lr")

plt.figure(figsize=(7, 5))
plt.errorbar(
    x["lr"],
    x["mean_steps"],
    yerr=x["std_steps"],
    marker="o",
    capsize=4
)
plt.xlabel("Learning rate")
plt.ylabel("Mean steps to convergence")
plt.title("Convergence Speed vs Learning Rate")
plt.xticks(x["lr"])
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f"{OUT}/9_convergence_steps.png", dpi=200)
plt.close()


print("Created 9 graphs in", OUT)