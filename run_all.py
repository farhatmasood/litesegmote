"""Run the experiment pipeline in order.

Assumes setup_upstream.py has been run and that the checkpoints and the two ISIC
archives are in place. Stages that already have their output are skipped, so the
pipeline can be resumed after an interruption.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def stage(name, marker, command):
    if os.path.exists(marker):
        print(f"[skip] {name}: {marker} exists")
        return
    print(f"\n=== {name} ===", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", default="third_party/segmote")
    parser.add_argument("--sam_checkpoint", default="checkpoints/sam_b.pth")
    parser.add_argument("--segmote_checkpoint", default="checkpoints/segmote.pth")
    parser.add_argument("--train_dir", default="datasets/isic2016_train")
    parser.add_argument("--test_dir", default="datasets/isic2016_test")
    parser.add_argument("--results", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    python = sys.executable
    scripts = os.path.join(HERE, "experiments")
    common = ["--segmote_root", args.segmote_root, "--sam_checkpoint", args.sam_checkpoint]
    weighted = common + ["--segmote_checkpoint", args.segmote_checkpoint]

    stage("parameter audit", os.path.join(args.results, "audit", "param_audit.json"),
          [python, os.path.join(scripts, "run_param_audit.py"), *common,
           "--out_dir", os.path.join(args.results, "audit")])

    stage("singular value spectra", os.path.join(args.results, "spectra", "energy_retention.json"),
          [python, os.path.join(scripts, "run_spectra.py"), *weighted,
           "--out_dir", os.path.join(args.results, "spectra")])

    stage("data-free sweep", os.path.join(args.results, "datafree", "datafree_sweep.json"),
          [python, os.path.join(scripts, "run_datafree_sweep.py"), *weighted,
           "--test_dir", args.test_dir, "--out_dir", os.path.join(args.results, "datafree")])

    stage("fine-tuning sweep",
          os.path.join(args.results, "finetune", f"summary_lite_seed{args.seeds[-1]}.json"),
          [python, os.path.join(scripts, "run_sweep.py"), *weighted,
           "--train_dir", args.train_dir, "--test_dir", args.test_dir,
           "--variants", "baseline", "lite",
           "--seeds", *[str(s) for s in args.seeds],
           "--epochs", str(args.epochs),
           "--out_dir", os.path.join(args.results, "finetune")])

    stage("statistics", os.path.join(args.results, "stats", "statistics.json"),
          [python, os.path.join(scripts, "run_statistics.py"),
           "--finetune_dir", os.path.join(args.results, "finetune"),
           "--out_dir", os.path.join(args.results, "stats")])

    stage("component ablation",
          os.path.join(args.results, "ablation", "component_ablation.json"),
          [python, os.path.join(scripts, "run_component_ablation.py"), *weighted,
           "--test_dir", args.test_dir, "--out_dir", os.path.join(args.results, "ablation")])

    stage("efficiency", os.path.join(args.results, "efficiency", "efficiency.json"),
          [python, os.path.join(scripts, "run_efficiency.py"), *weighted,
           "--test_dir", args.test_dir, "--out_dir", os.path.join(args.results, "efficiency")])

    stage("excitation-weighted spectra",
          os.path.join(args.results, "excitation", "excitation_summary.json"),
          [python, os.path.join(scripts, "run_excitation.py"), *weighted,
           "--test_dir", args.test_dir, "--out_dir", os.path.join(args.results, "excitation")])

    print("\nall stages complete")


if __name__ == "__main__":
    main()
