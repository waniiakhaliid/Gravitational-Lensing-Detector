"""
generators/image_generator.py
Single-image generator that wires together the physics engine and noise pipeline.
Called by the parallel batch runner.

Pipeline order (lensed images)
-------------------------------
  1. Sample intended_morph
  2. Sample PSF FWHM (fixed across retries for the same job)
  3. Retry loop (max MAX_RETRIES):
       a. Sample physics params (lens, source, optional subhalo)
       b. render_source_only_image()   ← NO PSF, NO lens light, NO noise
       c. analyze_morphology()
       d. Break if final_morph ∉ {ambiguous, no_lens}; else retry
  4. After loop:
       - Resolved  → render full image (PSF + lens light) → apply noise → save to images/
       - Ambiguous → render full image from last params   → apply noise → save to ambiguous/
  5. Return metadata row with ALL new fields
"""

import os
import numpy as np
from PIL import Image
from typing import Dict, Optional

from config import (
    IMG_SIZE, PIXEL_SCALE, PSF_P, MORPH_SPLIT, REALISM_SPLIT,
    MAX_RETRIES, AMBIGUOUS_DIR, ANALYSIS_VERSION,
)
from core.lensing_engine import (
    sample_physics_params,
    render_source_only_image,
    render_lensed_image_from_params,
    render_no_lens_image,
)
from core.morphology_analyzer import analyze_morphology
from core.noise_engine import apply_noise_and_realism


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLERS
# ─────────────────────────────────────────────────────────────────────────────

MORPH_LABELS = list(MORPH_SPLIT.keys())
MORPH_PROBS  = np.array(list(MORPH_SPLIT.values()), dtype=np.float64)
MORPH_PROBS /= MORPH_PROBS.sum()

REALISM_LABELS = list(REALISM_SPLIT.keys())
REALISM_PROBS  = np.array(list(REALISM_SPLIT.values()), dtype=np.float64)
REALISM_PROBS /= REALISM_PROBS.sum()


def sample_morphology(rng: np.random.Generator) -> str:
    return rng.choice(MORPH_LABELS, p=MORPH_PROBS)


def sample_realism(rng: np.random.Generator) -> str:
    return rng.choice(REALISM_LABELS, p=REALISM_PROBS)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lens_center_pixels(params: Dict) -> tuple:
    """
    Convert the lenstronomy sky-coordinate lens centre to pixel (row, col).

    lenstronomy convention (with inverse=True in make_grid_with_coordtransform):
      x  → column  (increases right)
      y  → row     (increases *up* in sky, so *decreases* in array row index)

    Approximate pixel position (exact for small offsets):
        row = IMG_SIZE/2 - cy / pixel_scale
        col = IMG_SIZE/2 + cx / pixel_scale
    """
    cx = params["kwargs_lens"][0]["center_x"]
    cy = params["kwargs_lens"][0]["center_y"]
    row = IMG_SIZE / 2.0 - cy / PIXEL_SCALE
    col = IMG_SIZE / 2.0 + cx / PIXEL_SCALE
    return (row, col)


def _ensure_size(img_u8: np.ndarray) -> np.ndarray:
    """Resize to IMG_SIZE × IMG_SIZE if the array is the wrong shape."""
    if img_u8.shape[0] != IMG_SIZE or img_u8.shape[1] != IMG_SIZE:
        img_u8 = np.array(
            Image.fromarray(img_u8).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
    return img_u8


def _save_png(img_u8: np.ndarray, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_u8, mode="L").save(filepath, format="PNG", optimize=False)


def _psf_focus_index(fwhm: float) -> float:
    """Normalised PSF sharpness: 0.0 = minimum FWHM (sharpest), 1.0 = maximum."""
    lo, hi = PSF_P.fwhm_range
    return round((fwhm - lo) / (hi - lo + 1e-9), 4)


def _psf_position_index(params: Dict) -> float:
    """Lens-centre offset from field centre in pixels (proxy for PSF field position)."""
    cx = params["kwargs_lens"][0]["center_x"]
    cy = params["kwargs_lens"][0]["center_y"]
    return round(np.sqrt(cx**2 + cy**2) / PIXEL_SCALE, 4)


# ─────────────────────────────────────────────────────────────────────────────
# CORE SINGLE-IMAGE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_single_image(
    idx         : int,
    main_class  : str,        # 'no_lens' | 'lens_no_subhalo' | 'lens_with_subhalo'
    realism     : str,        # 'clean' | 'semi_messy' | 'messy'
    output_dir  : str,
    seed        : int,
    morph       : Optional[str] = None,   # intended morphology; None → sampled randomly
) -> Dict:
    """
    Generate and save a single PNG image.

    For lensed images the pipeline is:
      sample intended_morph → retry loop (sample params → source-only render →
      analyze_morphology) → full render → noise → save.

    Returns a metadata dict (one row for the master CSV).
    The '_is_ambiguous' key is set to True for images routed to AMBIGUOUS_DIR;
    batch_runner uses this to exclude them from the main count.
    """
    rng = np.random.default_rng(seed)

    # ── Derived flags ─────────────────────────────────────────────────────────
    is_lens    = main_class in ("lens_no_subhalo", "lens_with_subhalo")
    with_sub   = (main_class == "lens_with_subhalo")
    lens_label = "lens"    if is_lens else "no_lens"
    sub_label  = "subhalo" if with_sub else "no_subhalo"

    # PSF FWHM is fixed for the job (same across retries) so the noise level
    # is consistent even when physics params are re-sampled.
    fwhm = rng.uniform(*PSF_P.fwhm_range)

    # ══════════════════════════════════════════════════════════════════════════
    # NO-LENS PATH  (skip morphology analysis entirely)
    # ══════════════════════════════════════════════════════════════════════════
    if not is_lens:
        clean_image, phys_meta = render_no_lens_image(rng=rng, fwhm_arcsec=fwhm)
        image_u8, noise_meta   = apply_noise_and_realism(clean_image, realism, rng, fwhm)
        image_u8 = _ensure_size(image_u8)

        filename = f"img_{idx:07d}.png"
        sub_dir  = os.path.join(output_dir, main_class, realism)
        _save_png(image_u8, os.path.join(sub_dir, filename))

        return {
            # ── Identifiers ──────────────────────────────────────────────────
            "filename"         : os.path.join(main_class, realism, filename),
            "idx"              : idx,
            "seed"             : seed,
            # ── Primary labels ───────────────────────────────────────────────
            "main_class"       : main_class,
            "lens_label"       : lens_label,
            "lens_type"        : "none",
            "subhalo_label"    : sub_label,
            "realism"          : realism,
            "split"            : "",
            # ── Morphology ───────────────────────────────────────────────────
            "intended_morph"            : "none",
            "final_morph"               : "no_lens",
            "num_components"            : 0,
            "angle_coverage_deg"        : 0.0,
            "mean_radius_pix"           : 0.0,
            "arc_pixels_count"          : 0,
            "theta_E_pix"               : 0.0,
            "einstein_ring_consistent"  : False,
            # ── PSF / source ─────────────────────────────────────────────────
            "psf_focus_index"   : _psf_focus_index(fwhm),
            "psf_position_index": 0.0,
            "source_size_pix"   : 0.0,
            # ── Ambiguous ────────────────────────────────────────────────────
            "ambiguous_reason"  : "",
            "analysis_version"  : ANALYSIS_VERSION,
            "_is_ambiguous"     : False,
            # ── Physics + noise ──────────────────────────────────────────────
            **phys_meta,
            **noise_meta,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # LENSED IMAGE PATH
    # ══════════════════════════════════════════════════════════════════════════

    # Step 1: Sample intended morphology (fixed for this job)
    intended_morph = morph if morph is not None else sample_morphology(rng)

    # State after the retry loop
    analysis_result   : Optional[Dict] = None
    final_params      : Optional[Dict] = None
    final_attempt_rng : np.random.Generator = rng
    last_result       : Optional[Dict] = None

    # Step 3: Retry loop
    for attempt in range(MAX_RETRIES):
        # Each attempt gets a fresh, reproducible sub-RNG derived from the job seed
        attempt_seed = int(rng.integers(0, 2**31))
        attempt_rng  = np.random.default_rng(attempt_seed)

        # 3a. Sample physics parameters
        params = sample_physics_params(attempt_rng, intended_morph, with_sub)

        # 3b. Render source-only image (NO PSF · NO lens light · NO noise)
        source_only = render_source_only_image(params)

        # 3c. Analyze morphology
        lens_center = _lens_center_pixels(params)
        result = analyze_morphology(
            source_only_img = source_only,
            theta_E         = params["theta_E"],
            pixel_scale     = PIXEL_SCALE,
            lens_center     = lens_center,
        )
        last_result = result

        # 3d. Check result quality
        # Retry on: ambiguous (uncertain boundary) or no_lens (arc too faint)
        if result["final_morph"] not in ("ambiguous", "no_lens"):
            analysis_result   = result
            final_params      = params
            final_attempt_rng = attempt_rng
            break   # ← good result, exit retry loop

    # ── Handle unresolved case ────────────────────────────────────────────────
    is_ambiguous = (analysis_result is None)
    if is_ambiguous:
        # Use last attempt's params/result for the saved image
        final_params      = params           # type: ignore[assignment]
        final_attempt_rng = attempt_rng      # type: ignore[possibly-undefined]
        analysis_result   = last_result      # type: ignore[assignment]
        ambiguous_reason  = (
            f"coverage={last_result['angle_coverage_deg']:.1f}deg,"    # type: ignore
            f"n_comp={last_result['num_components']},"
            f"flag={last_result['flag']}"
        )
    else:
        ambiguous_reason = ""

    # ── Step 4a: Render full image (PSF + lens light) ─────────────────────────
    # Never applied before this point — analysis was on the clean source-only image
    clean_image, phys_meta = render_lensed_image_from_params(final_params, fwhm)

    # ── Step 4b: Apply noise + realism ────────────────────────────────────────
    image_u8, noise_meta = apply_noise_and_realism(
        clean_image   = clean_image,
        realism_level = realism,
        rng           = final_attempt_rng,
        fwhm_arcsec   = fwhm,
    )
    image_u8 = _ensure_size(image_u8)

    # ── Step 4c: Save to correct folder ───────────────────────────────────────
    filename = f"img_{idx:07d}.png"
    if is_ambiguous:
        sub_dir  = os.path.join(AMBIGUOUS_DIR, main_class, realism)
        rel_path = os.path.join("ambiguous", main_class, realism, filename)
    else:
        sub_dir  = os.path.join(output_dir, main_class, realism)
        rel_path = os.path.join(main_class, realism, filename)

    _save_png(image_u8, os.path.join(sub_dir, filename))

    # ── Derived metadata ──────────────────────────────────────────────────────
    final_morph     = analysis_result["final_morph"]
    source_size_pix = round(
        final_params["meta_src"].get("src_R_sersic", 0.0) / PIXEL_SCALE, 4
    )

    return {
        # ── Identifiers ──────────────────────────────────────────────────────
        "filename"      : rel_path,
        "idx"           : idx,
        "seed"          : seed,
        # ── Primary labels ───────────────────────────────────────────────────
        "main_class"    : main_class,
        "lens_label"    : lens_label,
        "lens_type"     : final_morph,   # backward-compat alias for final_morph
        "subhalo_label" : sub_label,
        "realism"       : realism,
        "split"         : "",
        # ── Morphology analysis ──────────────────────────────────────────────
        "intended_morph"           : intended_morph,
        "final_morph"              : final_morph,
        "num_components"           : analysis_result["num_components"],
        "angle_coverage_deg"       : analysis_result["angle_coverage_deg"],
        "mean_radius_pix"          : analysis_result["mean_radius_pix"],
        "arc_pixels_count"         : analysis_result["arc_pixels_count"],
        "theta_E_pix"              : analysis_result["theta_E_pix"],
        "einstein_ring_consistent" : analysis_result["einstein_ring_consistent"],
        # ── PSF / source ─────────────────────────────────────────────────────
        "psf_focus_index"          : _psf_focus_index(fwhm),
        "psf_position_index"       : _psf_position_index(final_params),
        "source_size_pix"          : source_size_pix,
        # ── Ambiguous bookkeeping ────────────────────────────────────────────
        "ambiguous_reason"         : ambiguous_reason,
        "analysis_version"         : ANALYSIS_VERSION,
        "_is_ambiguous"            : is_ambiguous,
        # ── Physics + noise ──────────────────────────────────────────────────
        **phys_meta,
        **noise_meta,
    }
