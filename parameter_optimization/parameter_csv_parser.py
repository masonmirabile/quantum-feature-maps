import pandas as pd
import numpy as np

INPUT = "parameter_results.csv"
OUTPUT = "parameter_results_stats.csv"

df = pd.read_csv(INPUT)

summaries = []

def add_summary(name, columns):
    if len(columns) == 0:
        groups = [((), df)]
    else:
        groups = df.groupby(columns, dropna=False)

    for values, data in groups:
        if len(columns) == 0:
            values = ()
        elif len(columns) == 1:
            values = (values,)

        row = {
            "summary": name,
            "n": len(data),
            "mean_eval_acc": data["eval_acc"].mean(),
            "std_eval_acc": data["eval_acc"].std(),
            "se_eval_acc": data["eval_acc"].std() / np.sqrt(len(data)),
            "min_eval_acc": data["eval_acc"].min(),
            "max_eval_acc": data["eval_acc"].max(),
            "mean_train_acc": data["train_acc"].mean(),
            "mean_steps": data["steps"].mean(),
            "std_steps": data["steps"].std(),
            "convergence_rate": data["converged"].mean()
        }

        for column, value in zip(columns, values):
            row[column] = value

        summaries.append(row)

# overall statistics
add_summary("overall", [])

# individual variables
add_summary("layers", ["layers"])
add_summary("learning_rate", ["lr"])
add_summary("feature_map", ["feature_map"])
add_summary("noise", ["noise"])
add_summary("dataset", ["dataset"])

# parameter interactions
add_summary("layers_x_learning_rate", ["layers", "lr"])
add_summary("feature_map_x_layers", ["feature_map", "layers"])
add_summary("feature_map_x_learning_rate", ["feature_map", "lr"])

# noise interactions
add_summary("feature_map_x_noise", ["feature_map", "noise"])
add_summary("layers_x_noise", ["layers", "noise"])
add_summary("learning_rate_x_noise", ["lr", "noise"])

# dataset interactions
add_summary("dataset_x_layers", ["dataset", "layers"])
add_summary("dataset_x_learning_rate", ["dataset", "lr"])
add_summary("dataset_x_feature_map", ["dataset", "feature_map"])

stats = pd.DataFrame(summaries)

# put identifying columns first
columns = [
    "summary",
    "dataset",
    "feature_map",
    "noise",
    "layers",
    "lr",
    "n",
    "mean_eval_acc",
    "std_eval_acc",
    "se_eval_acc",
    "min_eval_acc",
    "max_eval_acc",
    "mean_train_acc",
    "mean_steps",
    "std_steps",
    "convergence_rate"
]

stats = stats.reindex(columns=columns)
stats.to_csv(OUTPUT, index=False)

print("Read", len(df), "runs")
print("Found", df["rep"].nunique(), "repetitions")
print("Reps:", sorted(df["rep"].unique()))
print("Created", len(stats), "summary rows")
print("Saved to", OUTPUT)