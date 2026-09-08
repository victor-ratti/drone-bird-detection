# -*- coding: utf-8 -*-
"""Download the Roboflow dataset into data/.

The API key is read from the .env file at the project root, never passed as a
command-line argument: arguments end up in the shell history. The .env file is
covered by .gitignore.

Usage:
    python src/download_data.py
"""

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(ROOT, ".env")
DEST = os.path.join(ROOT, "data")

WORKSPACE = "myworkspace-0p4nk"
PROJECT = "drone-bird-detection-3nl79"
VERSION = 3
FORMAT = "yolov11"


def read_env(path):
    """Minimal .env reader, to avoid depending on python-dotenv."""
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            values[key.strip()] = val.strip().strip('"').strip("'")
    return values


def main():
    key = os.environ.get("ROBOFLOW_API_KEY") or read_env(ENV_FILE).get("ROBOFLOW_API_KEY")
    if not key:
        print("API key missing.", file=sys.stderr)
        print(f"Set ROBOFLOW_API_KEY in {ENV_FILE}", file=sys.stderr)
        return 1

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Missing package. Run:", file=sys.stderr)
        print("  python -m pip install roboflow", file=sys.stderr)
        return 1

    os.makedirs(DEST, exist_ok=True)
    target = os.path.join(DEST, f"Drone-Bird-Detection-{VERSION}")
    if os.path.exists(os.path.join(target, "data.yaml")):
        print(f"Already present: {target}")
        return 0

    print(f"Downloading {PROJECT} v{VERSION} in {FORMAT} format")
    rf = Roboflow(api_key=key)
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    dataset = version.download(FORMAT, location=target)

    print()
    print("Downloaded to:", dataset.location)
    for split in ("train", "valid", "test"):
        n = len(glob.glob(os.path.join(dataset.location, split, "images", "*")))
        print(f"  {split:6}: {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
