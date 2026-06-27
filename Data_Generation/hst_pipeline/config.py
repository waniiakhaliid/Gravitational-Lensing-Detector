"""
config.py — Master configuration for HST-like gravitational lensing dataset.
All knobs live here. Edit this file to change dataset size, paths, or physics.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = "/content/hst_lensing_dataset"
IMAGES_DIR      = os.path.join(ROOT_DIR, "images")
METADATA_PATH   = os.path.join(ROOT_DIR, "metadata.csv")
SPLITS_DIR      = os.path.join(ROOT_DIR, "splits")
MODEL2_DIR      = os.path.join(ROOT_DIR, "model_2_lens_type")  # symlink-style flat copies

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROPERTIES — HST ACS/WFC-like
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE        = 128          # pixels (128×128)
PIXEL_SCALE     = 0.05         # arcsec/pixel  (ACS/WFC native ~0.05")
SUPERSAMPLE     = 3            # internal supersampling factor (renders at 3× then downsample)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET COUNTS — 100,000 total
# ─────────────────────────────────────────────────────────────────────────────
# Main class counts
CLASS_COUNTS: Dict[str, int] = {
    "no_lens"           : 50_000,
    "lens_no_subhalo"   : 25_000,
    "lens_with_subhalo" : 25_000,
}
TOTAL_IMAGES = sum(CLASS_COUNTS.values())   # 100,000

# Morphology distribution (applied to BOTH lens classes equally)
# Counts below refer to per-class totals that have lensing
MORPH_SPLIT: Dict[str, float] = {
    "ring"         : 0.20,
    "arc"          : 0.20,
    "double"       : 0.20,
    "quad"         : 0.20,
    "partial_ring" : 0.20,
}

# Realism level distribution (applied to ALL classes uniformly)
REALISM_SPLIT: Dict[str, float] = {
    "clean"      : 0.20,   # 20 000 images
    "semi_messy" : 0.30,   # 30 000 images
    "messy"      : 0.50,   # 50 000 images
}

# Train / Val / Test split fractions
SPLIT_FRACS: Dict[str, float] = {
    "train" : 0.70,
    "val"   : 0.15,
    "test"  : 0.15,
}

# ─────────────────────────────────────────────────────────────────────────────
# PARALLELISM
# ─────────────────────────────────────────────────────────────────────────────
NUM_WORKERS  = 4      # parallel worker processes for image generation
BATCH_SIZE   = 500    # images per worker batch (memory management)

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEED
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS PARAMETER RANGES
# All ranges are (min, max) — sampled uniformly unless noted
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LensParams:
    """Ranges for the main deflector (lens galaxy)."""
    # SIE Einstein radius [arcsec]
    theta_E_range:        Tuple[float, float] = (0.5, 2.0)
    # Ellipticity (e1, e2) component range
    ellip_range:          Tuple[float, float] = (-0.3, 0.3)
    # External shear magnitude
    gamma_ext_range:      Tuple[float, float] = (0.0, 0.1)
    # Lens galaxy effective radius [arcsec]  (Sersic)
    R_sersic_range:       Tuple[float, float] = (0.3, 1.2)
    # Sersic index for lens galaxy
    n_sersic_lens_range:  Tuple[float, float] = (3.0, 5.0)   # early-type
    # Amplitude range (counts, arbitrary)
    amp_lens_range:       Tuple[float, float] = (500, 3000)
    # Lens galaxy center offset from field center [arcsec]
    center_offset_range:  Tuple[float, float] = (-0.2, 0.2)

@dataclass
class SourceParams:
    """Ranges for the background source galaxy."""
    # Source position relative to lens [arcsec]
    beta_range:           Tuple[float, float] = (-0.3, 0.3)
    # Source effective radius [arcsec]  (Sersic)
    R_sersic_range:       Tuple[float, float] = (0.05, 0.3)
    # Sersic index for source
    n_sersic_src_range:   Tuple[float, float] = (0.5, 4.0)
    # Amplitude range (controls brightness of arcs)
    amp_src_range:        Tuple[float, float] = (50, 600)
    # Source ellipticity
    ellip_src_range:      Tuple[float, float] = (-0.4, 0.4)

@dataclass
class SubhaloParams:
    """
    NFW or point-mass subhalo parameters.
    Subhalos should create SUBTLE arc perturbations — not visible blobs.
    """
    # Subhalo mass [M_sun] — kept low for subtlety
    mass_range:           Tuple[float, float] = (1e8, 5e9)
    # Concentration parameter (NFW)
    concentration_range:  Tuple[float, float] = (5.0, 30.0)
    # Subhalo position (polar): radial distance from lens center [arcsec]
    r_range:              Tuple[float, float] = (0.1, 0.8)   # close to arc
    # Subhalo position: angular [deg]
    theta_range:          Tuple[float, float] = (0, 360)
    # Subhalo redshift (same as lens plane for simplicity)
    # Scale radius [arcsec] for NFW  (derived from mass+concentration)

@dataclass
class MorphologyConstraints:
    """
    Per-morphology beta (source position) constraints.
    These guide what arc morphology is produced.
    """
    # Source must be within this fraction of theta_E from lens center
    ring:         Tuple[float, float] = (0.00, 0.10)  # nearly on caustic
    partial_ring: Tuple[float, float] = (0.10, 0.20)
    arc:          Tuple[float, float] = (0.15, 0.40)
    double:       Tuple[float, float] = (0.30, 0.70)
    quad:         Tuple[float, float] = (0.05, 0.30)  # needs ellipticity

@dataclass
class NoLensParams:
    """Background / isolated galaxy field (no lensing)."""
    # Number of field galaxies
    n_galaxies_range:     Tuple[int,   int  ] = (1, 4)
    R_sersic_range:       Tuple[float, float] = (0.1, 1.5)
    n_sersic_range:       Tuple[float, float] = (0.5, 4.0)
    amp_range:            Tuple[float, float] = (20, 800)
    ellip_range:          Tuple[float, float] = (-0.5, 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# PSF PARAMETERS — HST ACS/WFC F814W-like
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PSFParams:
    # Gaussian PSF FWHM [arcsec]  (ACS/WFC ~0.10")
    fwhm_range:           Tuple[float, float] = (0.08, 0.13)
    # Optional tiny offset to simulate mild PSF ellipticity
    psf_e_range:          Tuple[float, float] = (0.00, 0.05)

# ─────────────────────────────────────────────────────────────────────────────
# NOISE & REALISM PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CleanNoiseParams:
    """Minimal noise — for Stage 1 training."""
    sky_background:       float = 5.0          # counts/pixel
    read_noise_sigma:     float = 3.0          # electrons (ACS ~3–5e-)
    exposure_time:        float = 2000.0       # seconds
    gain:                 float = 2.0          # e-/ADU
    # Small photometric scatter
    flux_jitter:          Tuple[float, float] = (0.95, 1.05)

@dataclass
class SemiMessyNoiseParams:
    """Moderate realism — PSF, noise, slight sky gradient, 1–2 faint extras."""
    sky_background:       float = 20.0
    read_noise_sigma:     float = 5.0
    exposure_time:        float = 1800.0
    gain:                 float = 2.0
    flux_jitter:          Tuple[float, float] = (0.90, 1.10)
    # Extra faint objects
    n_extra_range:        Tuple[int, int]     = (1, 2)
    extra_amp_range:      Tuple[float, float] = (5, 80)
    # Mild sky gradient (amplitude fraction)
    sky_gradient_amp:     float = 0.10

@dataclass
class MessyNoiseParams:
    """Full realism — HST-like, closest to real science data."""
    sky_background:       float = 50.0
    read_noise_sigma:     float = 7.0
    exposure_time:        float = 1400.0
    gain:                 float = 2.0
    flux_jitter:          Tuple[float, float] = (0.80, 1.20)
    # Background galaxies and stars
    n_bg_galaxies_range:  Tuple[int, int]     = (2, 6)
    bg_amp_range:         Tuple[float, float] = (5, 150)
    n_stars_range:        Tuple[int, int]     = (0, 3)
    star_amp_range:       Tuple[float, float] = (20, 400)
    # Sky gradient (can be up to 20% variation across frame)
    sky_gradient_amp:     float = 0.20
    # Mild cosmic ray / hot pixel rate (fraction of pixels)
    cr_fraction:          float = 0.0003
    # Vignetting: mild brightness roll-off at corners
    vignette_strength:    float = 0.08

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION (tiny geometric/photometric jitter — NOT heavy augmentation)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AugmentParams:
    rotation_range:       Tuple[float, float] = (-180, 180)   # deg  (full rotation)
    crop_fraction_range:  Tuple[float, float] = (0.0, 0.03)   # fraction of size
    intensity_jitter:     Tuple[float, float] = (0.95, 1.05)  # global flux scale

# ─────────────────────────────────────────────────────────────────────────────
# INSTANTIATED DEFAULTS (used by generators)
# ─────────────────────────────────────────────────────────────────────────────
LENS_P    = LensParams()
SRC_P     = SourceParams()
SUB_P     = SubhaloParams()
MORPH_C   = MorphologyConstraints()
NO_LENS_P = NoLensParams()
PSF_P     = PSFParams()
AUGMENT_P = AugmentParams()

NOISE_PARAMS = {
    "clean"      : CleanNoiseParams(),
    "semi_messy" : SemiMessyNoiseParams(),
    "messy"      : MessyNoiseParams(),
}
