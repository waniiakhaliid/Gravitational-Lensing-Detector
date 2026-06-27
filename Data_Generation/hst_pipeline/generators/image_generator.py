"""
generators/image_generator.py
Single-image generator that wires together the physics engine and noise pipeline.
Called by the parallel batch runner.
"""

import numpy as np
import os
from PIL import Image
from typing import Dict, Tuple, Optional

from config import (
    IMG_SIZE, PIXEL_SCALE, PSF_P, MORPH_SPLIT,
    REALISM_SPLIT,
)
from core.lensing_engine import (
    render_lensed_image,
    render_no_lens_image,
)
from core.noise_engine import apply_noise_and_realism


# ─────────────────────────────────────────────────────────────────────────────
# MORPHOLOGY SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

MORPH_LABELS = list(MORPH_SPLIT.keys())
MORPH_PROBS  = np.array(list(MORPH_SPLIT.values()), dtype=np.float64)
MORPH_PROBS /= MORPH_PROBS.sum()   # normalise


def sample_morphology(rng: np.random.Generator) -> str:
    return rng.choice(MORPH_LABELS, p=MORPH_PROBS)


# ─────────────────────────────────────────────────────────────────────────────
# REALISM SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

REALISM_LABELS = list(REALISM_SPLIT.keys())
REALISM_PROBS  = np.array(list(REALISM_SPLIT.values()), dtype=np.float64)
REALISM_PROBS /= REALISM_PROBS.sum()


def sample_realism(rng: np.random.Generator) -> str:
    return rng.choice(REALISM_LABELS, p=REALISM_PROBS)


# ─────────────────────────────────────────────────────────────────────────────
# CORE SINGLE-IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_image(
    idx         : int,
    main_class  : str,        # 'no_lens' | 'lens_no_subhalo' | 'lens_with_subhalo'
    realism     : str,        # 'clean' | 'semi_messy' | 'messy'
    output_dir  : str,
    seed        : int,
    morph       : Optional[str] = None,   # if None, sampled randomly
) -> Dict:
    """
    Generate and save a single PNG image.

    Returns a metadata dict (one row for the master CSV).
    """
    rng = np.random.default_rng(seed)

    # ── Derived labels ────────────────────────────────────────────────────────
    is_lens = main_class in ("lens_no_subhalo", "lens_with_subhalo")
    with_sub= (main_class == "lens_with_subhalo")

    lens_label   = "lens"    if is_lens else "no_lens"
    subhalo_label= "subhalo" if with_sub else "no_subhalo"
    lens_type    = "none"

    # Morphology only for lensed images
    if is_lens:
        lens_type = morph if morph is not None else sample_morphology(rng)

    # ── PSF ──────────────────────────────────────────────────────────────────
    fwhm = rng.uniform(*PSF_P.fwhm_range)

    # ── Physics simulation ───────────────────────────────────────────────────
    if is_lens:
        clean_image, phys_meta = render_lensed_image(
            rng         = rng,
            morph       = lens_type,
            with_subhalo= with_sub,
            fwhm_arcsec = fwhm,
        )
    else:
        clean_image, phys_meta = render_no_lens_image(
            rng         = rng,
            fwhm_arcsec = fwhm,
        )

    # ── Noise + realism ──────────────────────────────────────────────────────
    image_u8, noise_meta = apply_noise_and_realism(
        clean_image   = clean_image,
        realism_level = realism,
        rng           = rng,
        fwhm_arcsec   = fwhm,
    )

    # ── Ensure correct size ──────────────────────────────────────────────────
    if image_u8.shape[0] != IMG_SIZE or image_u8.shape[1] != IMG_SIZE:
        pil_img = Image.fromarray(image_u8).resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR
        )
        image_u8 = np.array(pil_img)

    # ── Build filename ────────────────────────────────────────────────────────
    # Format: {class}/{realism}/img_{idx:07d}.png
    sub_dir  = os.path.join(output_dir, main_class, realism)
    os.makedirs(sub_dir, exist_ok=True)
    filename = f"img_{idx:07d}.png"
    filepath = os.path.join(sub_dir, filename)

    # Save as grayscale PNG
    Image.fromarray(image_u8, mode="L").save(filepath, format="PNG", optimize=False)

    # ── Metadata row ──────────────────────────────────────────────────────────
    rel_path = os.path.join(main_class, realism, filename)

    meta_row = {
        # --- Identifiers ---
        "filename"     : rel_path,
        "idx"          : idx,
        "seed"         : seed,
        # --- Primary labels ---
        "main_class"   : main_class,
        "lens_label"   : lens_label,
        "lens_type"    : lens_type,
        "subhalo_label": subhalo_label,
        # --- Realism ---
        "realism"      : realism,
        # --- Split (filled later) ---
        "split"        : "",
        # --- Physics ---
        **phys_meta,
        # --- Noise ---
        **noise_meta,
    }
    return meta_row
