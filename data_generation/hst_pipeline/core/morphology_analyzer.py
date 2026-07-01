"""
core/morphology_analyzer.py
Morphology classification of gravitational lensing arcs.

Called on a **clean source-only image** (no PSF, no noise, no lens light)
immediately after ray-tracing and before any instrumental degradation.
This guarantees the label reflects the physical arc geometry, not artefacts.

Pipeline position
-----------------
  sample params
      ↓
  render_source_only_image()   ← no PSF, no lens light
      ↓
  analyze_morphology()          ← THIS MODULE
      ↓
  get final_morph
      ↓
  render full image + apply noise
      ↓
  save labeled as final_morph
"""

import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from typing import Tuple, Dict

from config import MIN_COMPONENT_AREA, ARC_PIXEL_FLOOR

# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION THRESHOLDS  [degrees of angular coverage]
# ─────────────────────────────────────────────────────────────────────────────
_THRESH_RING         = 270.0   # coverage > 270  → ring
_THRESH_PARTIAL_RING = 180.0   # 180 ≤ coverage ≤ 270 → partial_ring
_THRESH_ARC_LOW      = 60.0    # 60  ≤ coverage < 180 → arc
_BOUNDARY_BUFFER     = 5.0     # ±5° ambiguity zone around each threshold

# All coverage boundaries subject to the buffer check
_COVERAGE_THRESHOLDS = [_THRESH_ARC_LOW, _THRESH_PARTIAL_RING, _THRESH_RING]


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _near_boundary(coverage: float) -> bool:
    """
    Return True if *coverage* falls within ±BOUNDARY_BUFFER degrees of any
    classification threshold (60°, 180°, 270°).
    Images near a boundary are classified as ambiguous to avoid mislabelling.
    """
    return any(abs(coverage - t) < _BOUNDARY_BUFFER for t in _COVERAGE_THRESHOLDS)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_morphology(
    source_only_img : np.ndarray,
    theta_E         : float,
    pixel_scale     : float,
    lens_center     : Tuple[float, float],
) -> Dict:
    """
    Classify the morphology of a gravitational lensing arc.

    Parameters
    ----------
    source_only_img : np.ndarray
        Float32 array (H × W) — raw lenstronomy output with **no PSF**,
        **no lens light**, and **no noise**.  Values are in raw counts.
    theta_E : float
        SIE Einstein radius in arcseconds (from the sampled physics params).
    pixel_scale : float
        Arcseconds per pixel (should match config.PIXEL_SCALE).
    lens_center : (row, col)
        Lens centre position in pixel coordinates (not sky coordinates).
        Typically computed as:
            row = IMG_SIZE/2 - lens_cy / pixel_scale
            col = IMG_SIZE/2 + lens_cx / pixel_scale

    Returns
    -------
    dict with the following keys:
        final_morph              : str   — classified morphology label
        num_components           : int   — number of distinct arc components
        angle_coverage_deg       : float — degrees of sky angle subtended by arc
        mean_radius_pix          : float — mean arc radius from lens centre (px)
        arc_pixels_count         : int   — total above-threshold arc pixels
        theta_E_pix              : float — Einstein radius in pixels
        einstein_ring_consistent : bool  — mean radius within 40% of theta_E_pix
        flag                     : str   — pipe-delimited diagnostic flags
    """
    img         = source_only_img.astype(np.float64)
    theta_E_pix = theta_E / pixel_scale

    # ── Step 1: Threshold at 3σ + clean small blobs ────────────────────────
    median = np.median(img)
    std    = np.std(img)
    binary = img > (median + 3.0 * std)

    binary = remove_small_objects(binary, min_size=MIN_COMPONENT_AREA)

    # ── Step 2: Early exit — too few arc pixels → classify as no_lens ──────
    arc_pixels_count = int(binary.sum())
    if arc_pixels_count < ARC_PIXEL_FLOOR:
        return {
            "final_morph"              : "no_lens",
            "num_components"           : 0,
            "angle_coverage_deg"       : 0.0,
            "mean_radius_pix"          : 0.0,
            "arc_pixels_count"         : arc_pixels_count,
            "theta_E_pix"              : round(theta_E_pix, 4),
            "einstein_ring_consistent" : False,
            "flag"                     : "low_signal",
        }

    # ── Step 3: Connected components ───────────────────────────────────────
    labeled      = label(binary)
    props        = regionprops(labeled)
    valid_props  = [p for p in props if p.area >= MIN_COMPONENT_AREA]
    num_components = len(valid_props)

    # ── Step 4: Angular coverage (1° bins around lens centre) ──────────────
    arc_rows, arc_cols = np.where(binary)
    cy_ref, cx_ref = lens_center            # (row, col) in pixel space
    dy = arc_rows - cy_ref
    dx = arc_cols - cx_ref

    # atan2 gives angle in (-180, 180]; shift to [0, 360)
    angles_deg = np.degrees(np.arctan2(dy, dx)) % 360.0
    angle_bins = np.clip(np.floor(angles_deg).astype(np.int32), 0, 359)
    angle_coverage_deg = float(len(np.unique(angle_bins)))  # occupied 1° bins

    # ── Step 5: Mean radius from lens centre ───────────────────────────────
    radii           = np.sqrt(dy**2.0 + dx**2.0)
    mean_radius_pix = float(np.mean(radii))

    # ── Step 6: Einstein radius consistency ────────────────────────────────
    deviation              = abs(mean_radius_pix - theta_E_pix) / (theta_E_pix + 1e-9)
    einstein_ring_consistent = deviation < 0.4

    # ── Steps 7 & 8: Classify with ±5° boundary buffer ────────────────────
    flag_parts: list[str] = []
    if not einstein_ring_consistent:
        flag_parts.append("einstein_inconsistent")

    if _near_boundary(angle_coverage_deg):
        final_morph = "ambiguous"
        flag_parts.append(f"near_threshold_{angle_coverage_deg:.1f}deg")
    elif angle_coverage_deg > _THRESH_RING:
        final_morph = "ring"
    elif angle_coverage_deg >= _THRESH_PARTIAL_RING:
        final_morph = "partial_ring"
    elif num_components >= 4:
        final_morph = "quad"
    elif num_components == 2:
        final_morph = "double"
    elif angle_coverage_deg >= _THRESH_ARC_LOW:
        final_morph = "arc"
    else:
        # Coverage below 60° and not 2 or 4 components — genuinely ambiguous
        final_morph = "ambiguous"
        flag_parts.append(f"low_coverage_{angle_coverage_deg:.1f}deg")

    return {
        "final_morph"              : final_morph,
        "num_components"           : num_components,
        "angle_coverage_deg"       : round(angle_coverage_deg, 2),
        "mean_radius_pix"          : round(mean_radius_pix, 3),
        "arc_pixels_count"         : arc_pixels_count,
        "theta_E_pix"              : round(theta_E_pix, 4),
        "einstein_ring_consistent" : einstein_ring_consistent,
        "flag"                     : "|".join(flag_parts),
    }
