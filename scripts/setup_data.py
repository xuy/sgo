"""
Download and cache the Nemotron-Personas-USA dataset.

Downloads 1M synthetic US personas (~2GB) from HuggingFace.
Default location: <project_root>/data/nemotron/
Only runs once — subsequent calls detect the cached dataset and skip.

Usage:
    uv run python scripts/setup_data.py
    uv run python scripts/setup_data.py --data-dir /custom/path
"""

import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "nemotron"


def setup(data_dir: Path = DEFAULT_DATA_DIR):
    if (data_dir / "dataset_info.json").exists():
        ds = load_from_disk(str(data_dir))
        print(f"Dataset already cached: {data_dir}")
        print(f"  {len(ds)} personas, {len(ds.column_names)} fields")
        return ds

    print("Downloading nvidia/Nemotron-Personas-USA (1M rows, ~2GB)...")
    print("This only needs to happen once.\n")

    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    data_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(data_dir))

    print(f"\nSaved to {data_dir}")
    print(f"  {len(ds)} personas, {len(ds.column_names)} fields")
    print(f"  Columns: {ds.column_names}")
    return ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    setup(args.data_dir)
