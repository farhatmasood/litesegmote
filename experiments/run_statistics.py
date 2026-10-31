"""Aggregate the fine-tuning runs and test the compressed model against the control.

Across seeds the summary reports the mean and sample standard deviation of test Dice.
Within a seed the two variants segment the same images, so their per-image scores are
paired; the Wilcoxon signed-rank test is applied to those pairs because Dice
distributions over a test set are skewed and bounded, which violates the assumptions
of a paired t test.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


def load_per_image(path):
    scores = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            scores[row["image"]] = float(row["dice"])
    return scores


def paired_test(reference_csv, candidate_csv):
    reference, candidate = load_per_image(reference_csv), load_per_image(candidate_csv)
    shared = sorted(set(reference) & set(candidate))
    a = np.array([reference[k] for k in shared])
    b = np.array([candidate[k] for k in shared])
    difference = b - a
    if np.allclose(difference, 0):
        return {"n": len(shared), "mean_difference": 0.0, "p_value": 1.0, "identical": True}
    statistic, p_value = stats.wilcoxon(b, a)
    return {
        "n": len(shared),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "candidate_better": int((difference > 0).sum()),
        "reference_better": int((difference < 0).sum()),
        "identical": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune_dir", required=True)
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--candidate", default="lite")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.finetune_dir, "summary_*.json"))):
        with open(path) as handle:
            record = json.load(handle)
        runs[record["variant"]].append(record)

    aggregate = {}
    for variant, records in sorted(runs.items()):
        box = np.array([r["test_box"] for r in records])
        free = np.array([r["test_prompt_free"] for r in records])
        aggregate[variant] = {
            "seeds": [r["seed"] for r in records],
            "trainable_params": records[0]["trainable_params"],
            "reduction": records[0]["reduction"],
            "box_mean": float(box.mean()), "box_std": float(box.std(ddof=1)) if len(box) > 1 else 0.0,
            "prompt_free_mean": float(free.mean()),
            "prompt_free_std": float(free.std(ddof=1)) if len(free) > 1 else 0.0,
            "selected_epochs": [r["selected_epoch"] for r in records],
        }
        print(f"{variant:10s} params {records[0]['trainable_params']/1e6:6.3f}M  "
              f"box {box.mean():.4f} +/- {aggregate[variant]['box_std']:.4f}  "
              f"prompt-free {free.mean():.4f} +/- {aggregate[variant]['prompt_free_std']:.4f}  "
              f"(n={len(records)} seeds)")

    tests = {}
    for mode in ("box", "promptfree"):
        per_seed = []
        for record in runs.get(args.candidate, []):
            seed = record["seed"]
            reference_csv = os.path.join(args.finetune_dir,
                                         f"test_{args.reference}_seed{seed}_{mode}.csv")
            candidate_csv = os.path.join(args.finetune_dir,
                                         f"test_{args.candidate}_seed{seed}_{mode}.csv")
            if os.path.exists(reference_csv) and os.path.exists(candidate_csv):
                result = paired_test(reference_csv, candidate_csv)
                result["seed"] = seed
                per_seed.append(result)
                print(f"[{mode}] seed {seed}: mean difference {result['mean_difference']:+.4f}, "
                      f"p = {result['p_value']:.4g}")
        tests[mode] = per_seed

    with open(os.path.join(args.out_dir, "statistics.json"), "w") as handle:
        json.dump({"aggregate": aggregate, "paired_tests": tests}, handle, indent=2)


if __name__ == "__main__":
    main()
