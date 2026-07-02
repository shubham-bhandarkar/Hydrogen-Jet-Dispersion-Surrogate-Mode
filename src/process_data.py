"""
Builds a machine learning dataset from the BLASTNet Lifted Hydrogen Jet
simulation data.

Workflow

1. Load X and Y grid
2. Detect available snapshots
3. Read CFD variables
4. Merge into one dataframe
5. Engineer features
6. Sample points
7. Save final dataset
"""

from pathlib import Path
import re
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm



# Configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GRID_DIR = PROJECT_ROOT / "grid"
FIELD_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data_processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SHAPE = (1600, 2000)

INPUT_FEATURES = [
    "RHO_kgm-3",
    "UX_ms-1",
    "UY_ms-1",
    "P_Pa",
    "T_K",
]

TARGET = "YH2"

SAMPLE_PER_SNAPSHOT = 2500
RANDOM_STATE = 42
DTYPE = np.float32



# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)



# Binary Reader


def read_binary_field(filepath: Path) -> np.ndarray:
    """Read one BLASTNet binary field, return flattened float32 array."""
    arr = np.fromfile(filepath, dtype=DTYPE)
    expected = GRID_SHAPE[0] * GRID_SHAPE[1]
    if arr.size != expected:
        raise ValueError(
            f"{filepath.name} contains {arr.size:,} values "
            f"(expected {expected:,})"
        )
    return arr



# Grid Loader

def load_grid():
    logger.info("Loading computational grid...")
    x = read_binary_field(GRID_DIR / "X_m.dat")
    y = read_binary_field(GRID_DIR / "Y_m.dat")
    logger.info("Grid loaded successfully.")
    return x, y



# Snapshot Discovery


def discover_snapshots():
    logger.info("Scanning snapshot files...")
    pattern = re.compile(r"_id(\d+)\.dat$")
    ids = set()
    for file in FIELD_DIR.glob(f"{TARGET}_id*.dat"):
        match = pattern.search(file.name)
        if match:
            ids.add(int(match.group(1)))
    snapshots = sorted(ids)
    logger.info("Found %d snapshots", len(snapshots))
    return snapshots


# Load One Snapshot


def load_snapshot(snapshot: int, x, y):
    data = {"X": x, "Y": y}
    variables = INPUT_FEATURES + [TARGET]
    for variable in variables:
        filename = FIELD_DIR / f"{variable}_id{snapshot:04d}.dat"
        data[variable] = read_binary_field(filename)
    return pd.DataFrame(data)



# Feature Engineering


def engineer_features(df: pd.DataFrame):
    logger.info("Engineering features...")
    df["velocity_mag"] = np.sqrt(df["UX_ms-1"]**2 + df["UY_ms-1"]**2)
    df["radius"] = np.sqrt(df["X"]**2 + df["Y"]**2)
    df["theta"] = np.arctan2(df["Y"], df["X"])
    df["log_radius"] = np.log1p(df["radius"])
    return df



# Clean Dataset

def clean_dataframe(df):
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    after = len(df)
    logger.info("Removed %d rows", before - after)
    # Optionally filter physically unrealistic values:
    # df = df[(df["RHO_kgm-3"] > 0) & (df["T_K"] > 0)]
    return df



# Random Sampling


def sample_dataframe(df):
    if len(df) <= SAMPLE_PER_SNAPSHOT:
        return df
    return df.sample(n=SAMPLE_PER_SNAPSHOT, random_state=RANDOM_STATE).reset_index(drop=True)



# Build Dataset

def build_dataset():
    logger.info("=" * 70)
    logger.info("Starting dataset creation...")
    logger.info("=" * 70)

    x, y = load_grid()
    snapshots = discover_snapshots()
    if not snapshots:
        raise RuntimeError("No snapshots found.")

    logger.info("Processing %d snapshots...", len(snapshots))
    all_samples = []

    for snapshot in tqdm(snapshots, desc="Snapshots"):
        try:
            logger.info("Reading snapshot %04d", snapshot)
            df = load_snapshot(snapshot, x, y)
            df = engineer_features(df)
            df = clean_dataframe(df)
            df = sample_dataframe(df)
            df["snapshot"] = snapshot
            all_samples.append(df)
        except Exception as e:
            logger.warning("Skipping snapshot %04d: %s", snapshot, e)

    if not all_samples:
        raise RuntimeError("No valid snapshots were processed.")

    logger.info("Concatenating snapshots...")
    final_df = pd.concat(all_samples, ignore_index=True)
    final_df = final_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    logger.info("Final dataset created.")
    return final_df



# Dataset Summary

def dataset_summary(df):
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 70)
    logger.info("Rows        : %d", len(df))
    logger.info("Columns     : %d", len(df.columns))
    logger.info("\nColumns:")
    for col in df.columns:
        logger.info("   %s", col)
    logger.info("\nMissing values:\n%s", df.isna().sum())
    logger.info("\nTarget statistics:\n%s", df[TARGET].describe())



# Save Dataset


def save_dataset(df):
    output_file = OUTPUT_DIR / "final_dataset.parquet"
    logger.info("Saving dataset...")
    df.to_parquet(output_file, index=False)
    logger.info("Saved to: %s", output_file.resolve())



# Main


def main():
    logger.info("")
    logger.info("=" * 70)
    logger.info("BLASTNet Hydrogen Jet Dataset Processing")
    logger.info("=" * 70)
    df = build_dataset()
    dataset_summary(df)
    save_dataset(df)
    logger.info("")
    logger.info("=" * 70)
    logger.info("Dataset successfully created.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()