"""
generators/batch_runner.py
Orchestrates generation of the full 100 000-image dataset using
multiprocessing. Each worker generates an independent batch with
a unique seed range, then writes PNG files and returns metadata rows.

Design:
  - Build a flat job list (idx, main_class, realism, seed, morph/None)
  - Shuffle so worker loads are balanced across classes/realism
  - Split into worker chunks
  - Pool.map → collect metadata rows
  - Write master metadata.csv and train/val/test splits
"""

import os
import sys
import csv
import json
import random
import time
import math
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent dir to path so imports work in Colab
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ROOT_DIR, IMAGES_DIR, METADATA_PATH, SPLITS_DIR, MODEL2_DIR,
    CLASS_COUNTS, MORPH_SPLIT, REALISM_SPLIT, SPLIT_FRACS,
    NUM_WORKERS, BATCH_SIZE, GLOBAL_SEED,
)
from generators.image_generator import generate_single_image


# ─────────────────────────────────────────────────────────────────────────────
# JOB LIST BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _assign_morphology(idx_in_class: int, main_class: str, rng) -> str:
    """Return a deterministic morphology label for lensed images."""
    if main_class == "no_lens":
        return "none"
    morph_labels = list(MORPH_SPLIT.keys())
    morph_probs  = np.array(list(MORPH_SPLIT.values()), dtype=np.float64)
    morph_probs /= morph_probs.sum()
    return rng.choice(morph_labels, p=morph_probs)


def _assign_realism(idx_in_class: int, total_in_class: int, rng) -> str:
    """
    Sample realism level for each image.
    Stratified so the global REALISM_SPLIT holds across each class.
    """
    realism_labels = list(REALISM_SPLIT.keys())
    realism_probs  = np.array(list(REALISM_SPLIT.values()), dtype=np.float64)
    realism_probs /= realism_probs.sum()
    return rng.choice(realism_labels, p=realism_probs)


def build_job_list(global_seed: int = GLOBAL_SEED) -> List[Dict]:
    """
    Build the complete flat list of image generation jobs.
    Each job is a dict:
      idx, main_class, realism, morph, seed
    """
    rng  = np.random.default_rng(global_seed)
    jobs = []
    idx  = 0

    for main_class, n_total in CLASS_COUNTS.items():
        for i in range(n_total):
            realism = _assign_realism(i, n_total, rng)
            morph   = _assign_morphology(i, main_class, rng)
            seed    = int(rng.integers(0, 2**31))
            jobs.append({
                "idx"       : idx,
                "main_class": main_class,
                "realism"   : realism,
                "morph"     : morph,
                "seed"      : seed,
            })
            idx += 1

    # Shuffle to balance worker loads across classes
    rng.shuffle(jobs)
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION (must be top-level for multiprocessing.Pool)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(job: Dict) -> Dict:
    """Process a single image generation job. Returns metadata row."""
    try:
        meta_row = generate_single_image(
            idx        = job["idx"],
            main_class = job["main_class"],
            realism    = job["realism"],
            output_dir = IMAGES_DIR,
            seed       = job["seed"],
            morph      = job["morph"] if job["morph"] != "none" else None,
        )
        return meta_row
    except Exception as e:
        # Return a sentinel error row — don't crash the whole pool
        return {
            "filename"    : f"ERROR_{job['idx']}",
            "idx"         : job["idx"],
            "seed"        : job["seed"],
            "main_class"  : job["main_class"],
            "lens_label"  : "ERROR",
            "lens_type"   : "ERROR",
            "subhalo_label": "ERROR",
            "realism"     : job["realism"],
            "split"       : "",
            "_error"      : str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

def assign_splits(df: pd.DataFrame, global_seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """
    Stratified split by (main_class, realism) to ensure each bucket is
    proportionally represented in train/val/test.
    """
    rng   = np.random.default_rng(global_seed + 1)
    split_col = np.empty(len(df), dtype=object)

    groups = df.groupby(["main_class", "realism"])
    for (cls, rl), grp in groups:
        indices   = grp.index.to_numpy()
        rng.shuffle(indices)
        n         = len(indices)
        n_train   = int(n * SPLIT_FRACS["train"])
        n_val     = int(n * SPLIT_FRACS["val"])

        split_col[indices[:n_train]]       = "train"
        split_col[indices[n_train:n_train+n_val]] = "val"
        split_col[indices[n_train+n_val:]] = "test"

    df["split"] = split_col
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 FOLDER STRUCTURE (symlink-style flat copies of lens images)
# ─────────────────────────────────────────────────────────────────────────────

def build_model2_folders(df: pd.DataFrame) -> None:
    """
    Create model_2_lens_type/{ring,arc,double,quad,partial_ring}/ subfolders.
    We create HARD SYMLINKS (or copies on systems without symlink support)
    pointing to the master images directory.
    Files are only linked — no data is duplicated in Colab's disk.
    """
    lens_df = df[df["lens_label"] == "lens"].copy()
    morph_types = list(MORPH_SPLIT.keys())

    for mt in morph_types:
        os.makedirs(os.path.join(MODEL2_DIR, mt), exist_ok=True)

    for _, row in tqdm(lens_df.iterrows(), total=len(lens_df),
                       desc="Building model_2_lens_type symlinks"):
        lt = row["lens_type"]
        if lt not in morph_types:
            continue
        src  = os.path.join(IMAGES_DIR, row["filename"])
        dst  = os.path.join(MODEL2_DIR, lt, os.path.basename(row["filename"]))
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# SPLIT FILE WRITER
# ─────────────────────────────────────────────────────────────────────────────

def write_split_files(df: pd.DataFrame) -> None:
    """
    Write separate CSV files for train/val/test splits.
    Also write per-model CSVs for convenience.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        split_df.to_csv(os.path.join(SPLITS_DIR, f"{split_name}.csv"),
                        index=False)

    # Model-specific CSVs (only include relevant label columns)
    # Model 1: lens vs no_lens
    m1_cols = ["filename", "lens_label", "realism", "split"]
    df[m1_cols].to_csv(os.path.join(SPLITS_DIR, "model1_lens_vs_nolens.csv"),
                       index=False)

    # Model 2: lens type (only lensed images)
    m2_df   = df[df["lens_label"] == "lens"].copy()
    m2_cols = ["filename", "lens_type", "realism", "split"]
    m2_df[m2_cols].to_csv(os.path.join(SPLITS_DIR, "model2_lens_type.csv"),
                           index=False)

    # Model 3: subhalo vs no subhalo (only lensed images)
    m3_cols = ["filename", "subhalo_label", "realism", "split"]
    m2_df[m3_cols].to_csv(os.path.join(SPLITS_DIR, "model3_subhalo.csv"),
                           index=False)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET STATS REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def print_dataset_stats(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print(" DATASET SUMMARY")
    print("="*60)
    print(f"\nTotal images: {len(df):,}")

    print("\n── By main_class ──")
    print(df["main_class"].value_counts().to_string())

    print("\n── By realism ──")
    print(df["realism"].value_counts().to_string())

    print("\n── By lens_type (lensed images only) ──")
    lens_df = df[df["lens_label"] == "lens"]
    print(lens_df["lens_type"].value_counts().to_string())

    print("\n── By split ──")
    print(df["split"].value_counts().to_string())

    print("\n── Realism × Class (cross-tab) ──")
    ct = pd.crosstab(df["main_class"], df["realism"])
    print(ct.to_string())

    errors = df[df.get("_error", pd.Series(dtype=str)).notna()]
    if "_error" in df.columns:
        err_count = df["_error"].notna().sum()
        if err_count > 0:
            print(f"\n⚠️  {err_count} images failed to generate (see _error column).")

    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(num_workers: int = NUM_WORKERS,
                 resume: bool = True) -> None:
    """
    Run the full dataset generation pipeline.

    Args:
        num_workers:  Number of parallel worker processes.
        resume:       If True, skip images that already exist on disk.
    """
    print("=" * 60)
    print("  HST Gravitational Lensing Dataset Generator")
    print("=" * 60)

    # Create dirs
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    # Build job list
    print("\n[1/5] Building job list...")
    jobs = build_job_list(GLOBAL_SEED)
    print(f"      Total jobs: {len(jobs):,}")

    # Filter already-done jobs if resuming
    if resume and os.path.exists(METADATA_PATH):
        done_df  = pd.read_csv(METADATA_PATH)
        done_idx = set(done_df["idx"].tolist())
        jobs     = [j for j in jobs if j["idx"] not in done_idx]
        print(f"      Resuming — {len(done_idx):,} already done, "
              f"{len(jobs):,} remaining.")
        existing_rows = done_df.to_dict("records")
    else:
        existing_rows = []

    if not jobs:
        print("      Nothing to do! All images already generated.")
        df = pd.read_csv(METADATA_PATH)
    else:
        # Run workers
        print(f"\n[2/5] Generating images with {num_workers} workers...")
        t0 = time.time()

        meta_rows = list(existing_rows)

        # Colab / single-process fallback if num_workers == 1
        if num_workers == 1:
            for job in tqdm(jobs, desc="Generating"):
                meta_rows.append(_worker_fn(job))
        else:
            with Pool(processes=num_workers) as pool:
                for row in tqdm(
                    pool.imap_unordered(_worker_fn, jobs, chunksize=BATCH_SIZE),
                    total=len(jobs), desc="Generating"
                ):
                    meta_rows.append(row)

        elapsed = time.time() - t0
        print(f"      Done in {elapsed/60:.1f} min  "
              f"({len(meta_rows)/elapsed:.1f} img/s)")

        # Build DataFrame
        print("\n[3/5] Building metadata DataFrame...")
        df = pd.DataFrame(meta_rows)
        df.sort_values("idx", inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Assign splits
    print("\n[4/5] Assigning train/val/test splits...")
    df = assign_splits(df, GLOBAL_SEED)
    df.to_csv(METADATA_PATH, index=False)
    print(f"      Saved: {METADATA_PATH}")

    # Write split files and model CSVs
    write_split_files(df)
    print(f"      Split files saved to: {SPLITS_DIR}/")

    # Build model_2 folder structure
    print("\n[5/5] Building model_2_lens_type/ folder structure...")
    build_model2_folders(df)
    print(f"      Saved: {MODEL2_DIR}/")

    # Print stats
    print_dataset_stats(df)

    # Save dataset config summary as JSON
    config_summary = {
        "total_images"  : len(df),
        "image_size"    : 128,
        "pixel_scale"   : 0.05,
        "class_counts"  : CLASS_COUNTS,
        "realism_split" : REALISM_SPLIT,
        "split_fracs"   : SPLIT_FRACS,
        "morph_split"   : MORPH_SPLIT,
        "global_seed"   : GLOBAL_SEED,
    }
    with open(os.path.join(ROOT_DIR, "dataset_config.json"), "w") as f:
        json.dump(config_summary, f, indent=2)

    print(f"\n✅  Dataset complete → {ROOT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_pipeline(num_workers=args.workers, resume=not args.no_resume)
