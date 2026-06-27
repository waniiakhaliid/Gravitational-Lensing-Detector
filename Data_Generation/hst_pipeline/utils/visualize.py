"""
utils/visualize.py
Preview and QC utilities for the generated dataset.

Functions:
  plot_sample_grid()      — grid of images by class × realism
  plot_class_breakdown()  — bar chart of class / realism / split counts
  inspect_metadata()      — rich describe() of physics parameters
  plot_arc_morphologies() — sample of each lens_type side by side
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from typing import Optional, List

from config import ROOT_DIR, IMAGES_DIR, MORPH_SPLIT


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_img(filepath: str) -> np.ndarray:
    return np.array(Image.open(filepath).convert("L"))


def _sample_rows(df: pd.DataFrame, filters: dict, n: int,
                 seed: int = 0) -> pd.DataFrame:
    """Filter df by dict of {col: value} and sample n rows."""
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= (df[col] == val)
    sub = df[mask]
    if len(sub) == 0:
        return sub
    return sub.sample(min(n, len(sub)), random_state=seed)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SAMPLE GRID  (class × realism)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_grid(
    metadata_csv : str = os.path.join(ROOT_DIR, "metadata.csv"),
    images_dir   : str = IMAGES_DIR,
    n_per_cell   : int = 3,
    split        : str = "train",
    save_path    : Optional[str] = None,
) -> None:
    """
    Plot a grid: rows = main_class, cols = realism level.
    Each cell shows n_per_cell example images.
    """
    df = pd.read_csv(metadata_csv)
    df = df[df["split"] == split]

    classes   = ["no_lens", "lens_no_subhalo", "lens_with_subhalo"]
    realism   = ["clean", "semi_messy", "messy"]
    n_cols    = len(realism) * n_per_cell
    n_rows    = len(classes)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.2),
        facecolor="#0d0d0d",
    )
    fig.suptitle("HST Lensing Dataset — Class × Realism Level",
                 color="white", fontsize=13, y=1.01)

    for r, cls in enumerate(classes):
        for c_idx, rl in enumerate(realism):
            rows_sampled = _sample_rows(df, {"main_class": cls, "realism": rl},
                                        n_per_cell, seed=r * 10 + c_idx)
            for k in range(n_per_cell):
                col = c_idx * n_per_cell + k
                ax  = axes[r, col] if n_rows > 1 else axes[col]
                ax.set_facecolor("#0d0d0d")
                ax.axis("off")

                if k < len(rows_sampled):
                    fp  = os.path.join(images_dir,
                                       rows_sampled.iloc[k]["filename"])
                    img = _load_img(fp)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)

                # Column header on first row
                if r == 0 and k == 1:
                    ax.set_title(rl.replace("_", "\n"), color="#aaaaaa",
                                 fontsize=8)
        # Row label
        axes[r, 0].set_ylabel(cls.replace("_", "\n"), color="#cccccc",
                               fontsize=7, rotation=90, va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 2. ARC MORPHOLOGY GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_arc_morphologies(
    metadata_csv : str = os.path.join(ROOT_DIR, "metadata.csv"),
    images_dir   : str = IMAGES_DIR,
    n_per_type   : int = 4,
    realism      : str = "messy",
    save_path    : Optional[str] = None,
) -> None:
    """
    Show sample images for each lens morphology type.
    Rows = morphology (ring/arc/double/quad/partial_ring),
    Cols = example images.
    """
    df       = pd.read_csv(metadata_csv)
    df       = df[(df["lens_label"] == "lens") & (df["realism"] == realism)]
    morph_types = list(MORPH_SPLIT.keys())

    fig, axes = plt.subplots(
        len(morph_types), n_per_type,
        figsize=(n_per_type * 2.0, len(morph_types) * 2.2),
        facecolor="#0d0d0d",
    )
    fig.suptitle(f"Lens Morphology Types — realism={realism}",
                 color="white", fontsize=12, y=1.01)

    for r, mt in enumerate(morph_types):
        rows_sampled = _sample_rows(df, {"lens_type": mt}, n_per_type, seed=r)
        for k in range(n_per_type):
            ax = axes[r, k]
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")
            if k < len(rows_sampled):
                fp  = os.path.join(images_dir, rows_sampled.iloc[k]["filename"])
                img = _load_img(fp)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[r, 0].set_ylabel(mt, color="#cccccc", fontsize=8, rotation=90,
                               va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASS / REALISM BREAKDOWN BARS
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_breakdown(
    metadata_csv: str = os.path.join(ROOT_DIR, "metadata.csv"),
    save_path   : Optional[str] = None,
) -> None:
    df = pd.read_csv(metadata_csv)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#111111")
    fig.suptitle("Dataset Composition", color="white", fontsize=12)

    palette = {"no_lens": "#4C9BE8", "lens_no_subhalo": "#E8904C",
               "lens_with_subhalo": "#6EE84C"}
    r_palette = {"clean": "#73C2FB", "semi_messy": "#FBA973", "messy": "#FB7373"}

    # ── Panel 1: main class ──
    ax = axes[0]
    ax.set_facecolor("#1a1a1a")
    vc = df["main_class"].value_counts()
    bars = ax.bar(vc.index, vc.values,
                  color=[palette.get(k, "#aaa") for k in vc.index])
    ax.set_title("By Main Class", color="white")
    ax.tick_params(colors="white"); ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", color="white",
                fontsize=7)

    # ── Panel 2: realism level ──
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    vc = df["realism"].value_counts()
    bars = ax.bar(vc.index, vc.values,
                  color=[r_palette.get(k, "#aaa") for k in vc.index])
    ax.set_title("By Realism Level", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", color="white",
                fontsize=7)

    # ── Panel 3: train/val/test split ──
    ax = axes[2]
    ax.set_facecolor("#1a1a1a")
    vc = df["split"].value_counts()
    bars = ax.bar(vc.index, vc.values, color=["#73C2FB", "#FBA973", "#FB7373"])
    ax.set_title("Train / Val / Test", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", color="white",
                fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. PHYSICS PARAMETER DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_physics_distributions(
    metadata_csv: str = os.path.join(ROOT_DIR, "metadata.csv"),
    save_path   : Optional[str] = None,
) -> None:
    """
    Plot histograms of key physics parameters for lensed images.
    """
    df      = pd.read_csv(metadata_csv)
    lens_df = df[df["lens_label"] == "lens"].copy()

    params = {
        "lens_theta_E" : "Einstein Radius θ_E [arcsec]",
        "src_bx"       : "Source Position β_x [arcsec]",
        "src_R_sersic" : "Source R_eff [arcsec]",
        "lens_R_sersic": "Lens R_eff [arcsec]",
        "fwhm_arcsec"  : "PSF FWHM [arcsec]",
        "src_amp"      : "Source Amplitude",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), facecolor="#111111")
    axes = axes.flatten()
    fig.suptitle("Physics Parameter Distributions (Lensed Images)",
                 color="white", fontsize=11)

    for ax, (col, label) in zip(axes, params.items()):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")

        if col in lens_df.columns:
            vals = pd.to_numeric(lens_df[col], errors="coerce").dropna()
            ax.hist(vals, bins=40, color="#4C9BE8", edgecolor="#222", alpha=0.85)
            ax.set_title(label, color="#cccccc", fontsize=8)
        else:
            ax.set_title(f"{label}\n(not in metadata)", color="#666", fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLEAN vs SUBHALO SIDE-BY-SIDE
# ─────────────────────────────────────────────────────────────────────────────

def plot_subhalo_comparison(
    metadata_csv : str = os.path.join(ROOT_DIR, "metadata.csv"),
    images_dir   : str = IMAGES_DIR,
    n_pairs      : int = 5,
    realism      : str = "messy",
    save_path    : Optional[str] = None,
) -> None:
    """
    Side-by-side: lens_no_subhalo | lens_with_subhalo (same morphology).
    Demonstrates how subtle the subhalo perturbation is.
    """
    df      = pd.read_csv(metadata_csv)
    df      = df[df["realism"] == realism]
    no_sub  = df[df["subhalo_label"] == "no_subhalo"]
    with_sub= df[df["subhalo_label"] == "subhalo"]

    # Match by lens_type
    morph_types = list(MORPH_SPLIT.keys())[:n_pairs]

    fig, axes = plt.subplots(n_pairs, 2,
                             figsize=(5, n_pairs * 2.5),
                             facecolor="#0d0d0d")
    fig.suptitle(f"No Subhalo vs With Subhalo ({realism})",
                 color="white", fontsize=11, y=1.01)

    axes[0, 0].set_title("No Subhalo", color="#73C2FB", fontsize=9)
    axes[0, 1].set_title("With Subhalo", color="#FB7373", fontsize=9)

    for r, mt in enumerate(morph_types):
        for col_idx, (sub_df, color) in enumerate([
            (no_sub[no_sub["lens_type"] == mt],   "#73C2FB"),
            (with_sub[with_sub["lens_type"] == mt],"#FB7373"),
        ]):
            ax = axes[r, col_idx]
            ax.set_facecolor("#0d0d0d")
            ax.axis("off")
            if len(sub_df) > 0:
                row = sub_df.sample(1, random_state=r).iloc[0]
                fp  = os.path.join(images_dir, row["filename"])
                img = _load_img(fp)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_ylabel(mt, color="#aaaaaa", fontsize=7,
                          rotation=90, va="center") if col_idx == 0 else None

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")
    plt.show()
