import csv
import os
import time
from data import DATASETS, get_train_eval_data
from vqc import train_vqc

FEATURE_MAPS = ("angle", "z", "zz")
NOISE_LEVELS = (1, 4, 7, 11)
LEARNING_RATES = (.02, .04, .06, .08)
LAYERS = (1, 2, 4, 6)
N = 500
FILE = "parameter_results.csv"
BASE_SEED = 5000

mode = input("Run by hours or repetitions? (h/r): ").lower()

if mode == "h":
    hours = float(input("Hours to run: "))
    end_time = time.time() + hours * 3600
    target_reps = None
else:
    target_reps = int(input("Target repetitions per condition: "))
    end_time = None

# load completed runs
done = set()

if os.path.exists(FILE):
    with open(FILE, newline="") as f:
        for r in csv.DictReader(f):
            done.add((
                int(r["rep"]),
                r["dataset"],
                int(r["noise"]),
                r["feature_map"],
                int(r["layers"]),
                float(r["lr"])
            ))
    print("Loaded", len(done), "completed runs")

fields = [
    "rep", "seed", "dataset", "noise", "feature_map",
    "layers", "lr", "train_acc", "eval_acc"
]

rep = 0

while True:
    # in repetition mode, stop once every condition has enough reps
    if target_reps is not None and rep >= target_reps:
        print("Target repetitions complete")
        break

    seed = BASE_SEED + rep

    for dataset in DATASETS:
        for noise in NOISE_LEVELS:
            X_train, X_eval, y_train, y_eval = get_train_eval_data(
                dataset, n=N, noise_level=noise, seed=seed
            )

            for feature_map in FEATURE_MAPS:
                for layers in LAYERS:
                    for lr in LEARNING_RATES:
                        key = (rep, dataset, noise, feature_map, layers, lr)

                        # skip anything already finished
                        if key in done:
                            continue

                        # stop between runs if the time limit is reached
                        if end_time is not None and time.time() >= end_time:
                            print("Time limit reached")
                            quit()

                        result = train_vqc(
                            X_train, y_train, X_eval, y_eval,
                            feature_map_name=feature_map,
                            layers=layers,
                            lr=lr,
                            steps=150,
                            seed=seed
                        )

                        row = {
                            "rep": rep,
                            "seed": seed,
                            "dataset": dataset,
                            "noise": noise,
                            "feature_map": feature_map,
                            "layers": layers,
                            "lr": lr,
                            "train_acc": result["train_acc"],
                            "eval_acc": result["eval_acc"]
                        }

                        # save every completed run immediately
                        with open(FILE, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fields)
                            if f.tell() == 0:
                                writer.writeheader()
                            writer.writerow(row)

                        done.add(key)

                        print(
                            "rep", rep,
                            dataset,
                            "noise", noise,
                            feature_map,
                            "layers", layers,
                            "lr", lr,
                            "acc", round(result["eval_acc"], 3)
                        )

    rep += 1