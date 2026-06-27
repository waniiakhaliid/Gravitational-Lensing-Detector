"""
utils/dataset_loader.py
PyTorch Dataset classes for loading the generated HST lensing dataset.

Supports:
  - Stage-aware realism filtering (Stage 1: clean only, Stage 2: all)
  - Per-model label selection (Model 1/2/3)
  - Augmentation toggle
  - Fast loading from pre-cached file list
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple, Dict, Callable

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import ROOT_DIR, IMAGES_DIR, SPLITS_DIR, IMG_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS  (string → int)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAPS: Dict[str, Dict[str, int]] = {
    # Model 1: binary lens detection
    "model1": {
        "no_lens": 0,
        "lens"   : 1,
    },
    # Model 2: lens morphology type
    "model2": {
        "ring"        : 0,
        "arc"         : 1,
        "double"      : 2,
        "quad"        : 3,
        "partial_ring": 4,
    },
    # Model 3: subhalo detection
    "model3": {
        "no_subhalo": 0,
        "subhalo"   : 1,
    },
}

# Which metadata column each model reads its label from
MODEL_LABEL_COL: Dict[str, str] = {
    "model1": "lens_label",
    "model2": "lens_type",
    "model3": "subhalo_label",
}

# Realism levels allowed per training stage
STAGE_REALISM: Dict[int, List[str]] = {
    1: ["clean"],
    2: ["clean", "semi_messy", "messy"],
    3: ["clean", "semi_messy", "messy"],  # Stage 3 = test on real data (separate)
}


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(augment: bool = True, img_size: int = IMG_SIZE) -> T.Compose:
    """
    Return torchvision transform pipeline.
    Images are grayscale uint8 PNGs → float tensor [0,1] single-channel.
    """
    ops = []
    ops.append(T.Grayscale(num_output_channels=1))     # ensure 1-channel
    ops.append(T.Resize((img_size, img_size)))

    if augment:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # Random rotation already baked into PNGs, but a small extra
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.05, contrast=0.05),
        ]

    ops += [
        T.ToTensor(),                                  # → [1, H, W] float [0,1]
    ]
    return T.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class HSTLensingDataset(Dataset):
    """
    PyTorch Dataset for HST gravitational lensing simulations.

    Args:
        model_name:   'model1' | 'model2' | 'model3'
        split:        'train' | 'val' | 'test'
        stage:        Training stage (1 = clean only, 2 = all realism levels).
        realism_levels: Override stage — pass explicit list of realism levels.
        transform:    Torchvision transform pipeline (defaults to get_transforms).
        augment:      Apply data augmentation (ignored if custom transform given).
        root_dir:     Root directory of the dataset.
        metadata_df:  Pre-loaded DataFrame (avoids re-reading CSV).
    """

    def __init__(
        self,
        model_name    : str,
        split         : str = "train",
        stage         : int = 1,
        realism_levels: Optional[List[str]] = None,
        transform     : Optional[Callable] = None,
        augment       : bool = True,
        root_dir      : str = ROOT_DIR,
        metadata_df   : Optional[pd.DataFrame] = None,
    ):
        assert model_name in LABEL_MAPS, f"model_name must be one of {list(LABEL_MAPS.keys())}"
        assert split in ("train", "val", "test"), "split must be 'train'|'val'|'test'"

        self.model_name   = model_name
        self.split        = split
        self.root_dir     = root_dir
        self.label_col    = MODEL_LABEL_COL[model_name]
        self.label_map    = LABEL_MAPS[model_name]
        self.images_dir   = os.path.join(root_dir, "images")

        # Load metadata
        if metadata_df is not None:
            df = metadata_df.copy()
        else:
            csv_path = os.path.join(root_dir, "metadata.csv")
            df = pd.read_csv(csv_path)

        # ── Filter by split ──────────────────────────────────────────────────
        df = df[df["split"] == split].copy()

        # ── Filter by realism level ──────────────────────────────────────────
        if realism_levels is not None:
            allowed = realism_levels
        else:
            allowed = STAGE_REALISM[stage]
        df = df[df["realism"].isin(allowed)].copy()

        # ── Model 2 + 3: only lensed images ─────────────────────────────────
        if model_name in ("model2", "model3"):
            df = df[df["lens_label"] == "lens"].copy()

        # ── Drop rows with missing labels ────────────────────────────────────
        df = df[df[self.label_col].isin(self.label_map.keys())].copy()
        df.reset_index(drop=True, inplace=True)

        self.df        = df
        self.transform = transform or get_transforms(
            augment=(augment and split == "train")
        )

        print(f"[HSTLensingDataset] model={model_name} | split={split} | "
              f"stage={stage} | n={len(self.df):,} | "
              f"realism={allowed} | classes={list(self.label_map.keys())}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        row      = self.df.iloc[i]
        img_path = os.path.join(self.images_dir, row["filename"])

        # Load grayscale PNG
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        label = self.label_map[row[self.label_col]]
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for weighted loss functions.
        Useful when classes are slightly imbalanced after filtering.
        """
        counts = self.df[self.label_col].value_counts()
        n_total= len(self.df)
        weights= torch.zeros(len(self.label_map))
        for label_str, label_int in self.label_map.items():
            if label_str in counts:
                weights[label_int] = n_total / (len(self.label_map) * counts[label_str])
        return weights

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def class_names(self) -> List[str]:
        return [k for k, _ in sorted(self.label_map.items(), key=lambda x: x[1])]


# ─────────────────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    model_name   : str,
    stage        : int = 1,
    batch_size   : int = 64,
    num_workers  : int = 2,
    root_dir     : str = ROOT_DIR,
    pin_memory   : bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for a given model and training stage.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    # Pre-load metadata once
    df = pd.read_csv(os.path.join(root_dir, "metadata.csv"))

    loaders = {}
    for split in ("train", "val", "test"):
        ds = HSTLensingDataset(
            model_name  = model_name,
            split       = split,
            stage       = stage,
            augment     = (split == "train"),
            root_dir    = root_dir,
            metadata_df = df,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),
        )

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(root_dir: str = ROOT_DIR) -> None:
    """Load one batch from each model/stage and verify shapes and labels."""
    import torch
    print("\n── Sanity check ──────────────────────────────────────────────")
    for model in ("model1", "model2", "model3"):
        for stage in (1, 2):
            try:
                loaders = make_dataloaders(
                    model_name  = model,
                    stage       = stage,
                    batch_size  = 8,
                    num_workers = 0,
                    root_dir    = root_dir,
                    pin_memory  = False,
                )
                imgs, labels = next(iter(loaders["train"]))
                print(f"  {model} stage={stage} train: "
                      f"imgs {tuple(imgs.shape)} | "
                      f"labels {labels.tolist()}")
            except Exception as e:
                print(f"  {model} stage={stage} ERROR: {e}")
    print("─"*60 + "\n")
