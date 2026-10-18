import csv
import time
from pathlib import Path

from data import DATASETS, get_train_eval_data
from vqc import FEATURE_MAPS, train_vqc

NOISE_LEVELS = range(1, 12)
REPS = 30
BASE_SEED = 10000
N = 500
LAYERS = 4
LR = 0.08
THRESHOLD = 1e-4
MAX_STEPS = 1000
FILE = Path(__file__).with_name("main_results.csv")

FIELDS = [
    "rep", "seed", "dataset", "noise", "feature_map", "layers", "lr",
    "train_acc", "eval_acc", "steps", "converged", "runtime_sec"
]


def main():
    mode = input("Run by hours or repetitions? (h/r): ").strip().lower()
    if mode == "h":
        max_hours = float(input("Hours to run: "))
        if max_hours <= 0:
            raise ValueError("Hours must be positive.")
        target_reps = REPS
    elif mode == "r":
        target_reps = int(input(f"Target total repetitions per condition (1-{REPS}): "))
        if not 1 <= target_reps <= REPS:
            raise ValueError(f"Repetitions must be between 1 and {REPS}.")
        max_hours = None
    else:
        raise ValueError("Choose 'h' or 'r'.")

    done = set()
    if FILE.exists():
        with FILE.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing results file has a different column format.")
            for row in reader:
                rep = int(row["rep"])
                if (int(row["seed"]) != BASE_SEED + rep or
                        int(row["layers"]) != LAYERS or float(row["lr"]) != LR):
                    raise ValueError("Existing results use different experiment settings.")
                key = (rep, row["dataset"], int(row["noise"]), row["feature_map"])
                if key in done:
                    raise ValueError(f"Duplicate result in CSV: {key}")
                done.add(key)

    total = target_reps * len(DATASETS) * len(NOISE_LEVELS) * len(FEATURE_MAPS)
    completed = sum(1 for rep, dataset, noise, name in done
                    if rep < target_reps and name in FEATURE_MAPS
                    and dataset in DATASETS and noise in NOISE_LEVELS)
    print(f"Completed: {completed}/{total}")
    start = time.monotonic()

    for rep in range(target_reps):
        seed = BASE_SEED + rep
        for dataset in DATASETS:
            for noise in NOISE_LEVELS:
                if all((rep, dataset, noise, name) in done for name in FEATURE_MAPS):
                    continue

                X_train, X_eval, y_train, y_eval = get_train_eval_data(
                    dataset, n=N, noise_level=noise, seed=seed
                )

                for name in FEATURE_MAPS:
                    key = (rep, dataset, noise, name)
                    if key in done:
                        continue
                    if max_hours is not None and (time.monotonic() - start) / 3600 >= max_hours:
                        print("Time limit reached. Run this file again to resume.")
                        return

                    print(f"Running {completed+1}/{total}: rep={rep}, {dataset}, noise={noise}, map={name}", flush=True)
                    t0 = time.monotonic()
                    result = train_vqc(
                        X_train, y_train, X_eval, y_eval,
                        feature_map_name=name, layers=LAYERS, lr=LR,
                        threshold=THRESHOLD, max_steps=MAX_STEPS, seed=seed
                    )
                    runtime = time.monotonic() - t0

                    row = {
                        "rep": rep, "seed": seed, "dataset": dataset,
                        "noise": noise, "feature_map": name,
                        "layers": LAYERS, "lr": LR,
                        "train_acc": result["train_acc"],
                        "eval_acc": result["eval_acc"],
                        "steps": result["steps"],
                        "converged": result["converged"],
                        "runtime_sec": round(runtime, 3)
                    }
                    with FILE.open("a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)

                    done.add(key)
                    completed += 1
                    print(f"  acc={result['eval_acc']:.3f}, steps={result['steps']}, time={runtime:.1f}s", flush=True)

    print(f"Complete: {completed}/{total} runs. Results saved to {FILE.name}")


if __name__ == "__main__":
    main()
