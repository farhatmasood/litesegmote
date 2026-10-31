"""Fetch the upstream SegMoTE sources at the exact commit used for the study.

The upstream project is not vendored into this repository. Cloning it here keeps
attribution clear and pins the model graph to a single revision, so the reported
parameter counts and scores refer to an unambiguous version of the code.
"""
import argparse
import os
import subprocess
import sys

UPSTREAM_URL = "https://github.com/InMyDreammer/SegMoTE.git"
UPSTREAM_COMMIT = "371502e369295cb98878a5c4227cb1c5f5c7c0ec"


def run(command, **kwargs):
    print("+", " ".join(command))
    result = subprocess.run(command, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="third_party/segmote")
    parser.add_argument("--commit", default=UPSTREAM_COMMIT)
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    if os.path.exists(os.path.join(dest, ".git")):
        print(f"{dest} already exists, fetching")
        run(["git", "-C", dest, "fetch", "--all", "--tags"])
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        run(["git", "clone", UPSTREAM_URL, dest])

    run(["git", "-C", dest, "checkout", "--detach", args.commit])
    print(f"\nupstream ready at {dest} (commit {args.commit})")
    print("pass this path to the experiment scripts with --segmote_root")


if __name__ == "__main__":
    sys.exit(main())
