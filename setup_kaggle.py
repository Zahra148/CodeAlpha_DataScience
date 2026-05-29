"""
============================================================
  KAGGLE SETUP & DATASET DOWNLOADER
  CodeAlpha Data Science Internship — All Tasks
============================================================

This script:
  1. Guides you to set up your Kaggle API credentials
  2. Auto-downloads datasets for all 4 tasks

HOW TO GET YOUR KAGGLE API KEY:
  1. Go to https://www.kaggle.com
  2. Click your profile picture → "Settings"
  3. Scroll to "API" section → Click "Create New Token"
  4. A file called kaggle.json is downloaded
  5. Place it at:
       - Linux/Mac : ~/.kaggle/kaggle.json
       - Windows   : C:\\Users\\<YourName>\\.kaggle\\kaggle.json
  6. Run this script: python setup_kaggle.py
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# ── Install kaggle package if missing ────────────────────────
try:
    import kaggle
except ImportError:
    print("[*] Installing kaggle package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    import kaggle

# ── Verify kaggle.json exists ─────────────────────────────────
kaggle_dir  = Path.home() / ".kaggle"
kaggle_json = kaggle_dir / "kaggle.json"

if not kaggle_json.exists():
    print("\n❌ kaggle.json NOT found!")
    print("   Please follow these steps:")
    print("   1. Go to https://www.kaggle.com/settings")
    print("   2. Under 'API', click 'Create New Token'")
    print("   3. Move the downloaded kaggle.json to:")
    print(f"      {kaggle_json}")
    print("\n   Then re-run this script.\n")
    sys.exit(1)

# Secure the file permissions (required by Kaggle API on Linux/Mac)
kaggle_json.chmod(0o600)
print(f"[✓] kaggle.json found at {kaggle_json}")

# ── Dataset definitions ───────────────────────────────────────
DATASETS = {
    "task1_iris": {
        "name"    : "Task 1 — Iris Flower Classification",
        "dataset" : "uciml/iris",
        "folder"  : "task1_iris",
    },
    "task2_unemployment": {
        "name"    : "Task 2 — Unemployment Analysis",
        "dataset" : "gokulrajkmv/unemployment-in-india",
        "folder"  : "task2_unemployment",
    },
    "task3_car_price": {
        "name"    : "Task 3 — Car Price Prediction",
        "dataset" : "vijayaadithyanvg/car-price-prediction",
        "folder"  : "task3_car_price",
    },
    "task4_sales": {
        "name"    : "Task 4 — Sales Prediction",
        "dataset" : "bumba12345/advertising",
        "folder"  : "task4_sales",
    },
}

# ── Download each dataset ─────────────────────────────────────
print("\n[*] Downloading all datasets...\n")

for key, info in DATASETS.items():
    dest = Path(info["folder"])
    dest.mkdir(exist_ok=True)

    print(f"  → {info['name']}")
    print(f"    Dataset : {info['dataset']}")
    print(f"    Saving  : ./{info['folder']}/")

    try:
        subprocess.check_call([
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", info["dataset"],
            "-p", str(dest),
            "--unzip",
            "--quiet",
        ])
        files = list(dest.glob("*.csv"))
        print(f"    ✓ Downloaded: {[f.name for f in files]}\n")
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Failed — check dataset slug or credentials: {e}\n")

print("=" * 55)
print("  All datasets downloaded. You can now run:")
print("    python task1_iris_classification.py")
print("    python task2_unemployment_analysis.py")
print("    python task3_car_price_prediction.py")
print("    python task4_sales_prediction.py")
print("=" * 55)
