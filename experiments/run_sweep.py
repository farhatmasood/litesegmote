"""Run the fine-tuning comparison across variants and seeds.

Each combination is a separate process so that a failure in one run cannot corrupt
the state of another. Results accumulate as JSON summaries in ``--out_dir``.
"""
import argparse
import itertools
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmote_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--segmote_checkpoint", required=True)
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["baseline", "lite"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--weights_dir", default=None)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(here, "run_finetune.py")
    os.makedirs(args.out_dir, exist_ok=True)

    for variant, seed in itertools.product(args.variants, args.seeds):
        summary = os.path.join(args.out_dir, f"summary_{variant}_seed{seed}.json")
        if os.path.exists(summary):
            print(f"[sweep] {variant} seed {seed} already complete, skipping")
            continue

        command = [
            sys.executable, script,
            "--segmote_root", args.segmote_root,
            "--sam_checkpoint", args.sam_checkpoint,
            "--segmote_checkpoint", args.segmote_checkpoint,
            "--train_dir", args.train_dir,
            "--test_dir", args.test_dir,
            "--variant", variant,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--out_dir", args.out_dir,
        ]
        if args.weights_dir and seed == args.seeds[0]:
            command += ["--save_weights",
                        os.path.join(args.weights_dir, f"segmote_{variant}_seed{seed}.pth")]

        print(f"[sweep] starting {variant} seed {seed}", flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"[sweep] {variant} seed {seed} failed with code {result.returncode}")
            return result.returncode
    print("[sweep] all runs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
