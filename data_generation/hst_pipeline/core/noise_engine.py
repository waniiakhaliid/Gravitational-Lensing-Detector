"""
core/noise_engine.py
Noise & realism pipeline: takes a clean ray-traced image and applies
physically motivated degradations to produce clean / semi_messy / messy output.

Order of operations (matches real CCD reduction pipeline in reverse):
  1. Scale to exposure time
  2. Add sky background (+ optional gradient)
  3. Apply Poisson noise (photon noise)
  4. Add read noise (Gaussian)
  5. Add cosmic rays / hot pixels        (messy only)
  6. Add background galaxies / stars     (semi_messy + messy)
  7. Apply vignetting                    (messy only)
  8. Apply minor photometric jitter
  9. Convolve with PSF (already done in lensing_engine — skip double-convolution)
 10. Normalise + convert to uint8 PNG-ready array
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Optional, Tuple, Dict
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF as LensPSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel as LensModelObj
import lenstronomy.Util.util as util

from config import (
    IMG_SIZE, PIXEL_SCALE, SUPERSAMPLE,
    NOISE_PARAMS, PSF_P, AUGMENT_P,
    NO_LENS_P,
)
from core.lensing_engine import make_psf_object, make_image_data


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sky_gradient(img_size: int, rng: np.random.Generator,
                  amplitude: float, base: float) -> np.ndarray:
    """
    Create a smooth sky-background gradient across the frame.
    amplitude: fraction of base sky level (e.g. 0.10 = ±10%)
    """
    angle   = rng.uniform(0, 2 * np.pi)
    cx, cy  = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    y, x    = np.mgrid[0:img_size, 0:img_size] / img_size
    gradient= np.cos(angle) * (x - cx) + np.sin(angle) * (y - cy)
    gradient= gradient / (gradient.max() - gradient.min() + 1e-9)
    return base + amplitude * base * gradient


def _add_bg_galaxy(canvas: np.ndarray, rng: np.random.Generator,
                   amp_range: Tuple[float, float],
                   pixel_scale: float = PIXEL_SCALE) -> None:
    """Inject a single faint background galaxy (Sersic) directly onto canvas."""
    img_size = canvas.shape[0]
    half     = (img_size * pixel_scale) / 2.0
    amp  = rng.uniform(*amp_range)
    R    = rng.uniform(0.05, 0.6)
    n    = rng.uniform(0.5, 3.0)
    e1   = rng.uniform(-0.4, 0.4)
    e2   = rng.uniform(-0.4, 0.4)
    cx   = rng.uniform(-half * 0.8, half * 0.8)
    cy   = rng.uniform(-half * 0.8, half * 0.8)

    # Minimal lenstronomy call for a single galaxy stamp
    psf_fwhm = rng.uniform(*PSF_P.fwhm_range)
    psf_obj  = make_psf_object(psf_fwhm, pixel_scale)
    data_obj = make_image_data(img_size, pixel_scale)
    lm       = LightModel(["SERSIC_ELLIPSE"])
    im_model = ImageModel(
        data_class=data_obj, psf_class=psf_obj,
        lens_model_class=LensModelObj([]),
        source_model_class=lm,
        kwargs_numerics={"supersampling_factor": 1,
                         "supersampling_convolution": False},
    )
    stamp = im_model.image(
        kwargs_lens=[], kwargs_ps=None,
        kwargs_lens_light=[],
        kwargs_source=[{"amp": amp, "R_sersic": R, "n_sersic": n,
                        "e1": e1, "e2": e2,
                        "center_x": cx, "center_y": cy}],
    )
    canvas += stamp.astype(np.float32)


def _add_star(canvas: np.ndarray, rng: np.random.Generator,
              amp_range: Tuple[float, float],
              fwhm_arcsec: float,
              pixel_scale: float = PIXEL_SCALE) -> None:
    """Inject a point-source star (PSF-convolved delta function)."""
    img_size  = canvas.shape[0]
    half_pix  = img_size // 2
    amp       = rng.uniform(*amp_range)
    px        = rng.integers(5, img_size - 5)
    py        = rng.integers(5, img_size - 5)

    # Gaussian PSF stamp
    sigma_pix = (fwhm_arcsec / pixel_scale) / (2 * np.sqrt(2 * np.log(2)))
    y, x      = np.mgrid[0:img_size, 0:img_size]
    star      = amp * np.exp(-((x - px)**2 + (y - py)**2) / (2 * sigma_pix**2))
    canvas   += star.astype(np.float32)


def _cosmic_rays(canvas: np.ndarray, rng: np.random.Generator,
                 fraction: float) -> None:
    """Randomly set a fraction of pixels to high values (cosmic ray hits)."""
    n_hits  = int(canvas.size * fraction)
    if n_hits == 0:
        return
    idx     = rng.choice(canvas.size, size=n_hits, replace=False)
    yy, xx  = np.unravel_index(idx, canvas.shape)
    # CR intensities: 2–10× local max (very hot pixels)
    cr_amp  = rng.uniform(canvas.max() * 2, canvas.max() * 10, size=n_hits)
    canvas[yy, xx] += cr_amp.astype(np.float32)


def _vignette(canvas: np.ndarray, strength: float) -> None:
    """Apply a mild radial brightness rolloff at frame corners (in-place)."""
    h, w    = canvas.shape
    cy, cx  = h / 2.0, w / 2.0
    y, x    = np.mgrid[0:h, 0:w]
    r       = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_norm  = r / r.max()
    factor  = 1.0 - strength * r_norm**2
    canvas *= factor.astype(np.float32)


def _normalise_to_uint8(image: np.ndarray,
                         low_pct: float = 0.5,
                         high_pct: float = 99.5) -> np.ndarray:
    """
    Normalise a float image to [0, 255] uint8 using percentile clipping.
    Mimics how HST drizzled images are displayed / saved for ML use.
    """
    lo  = np.percentile(image, low_pct)
    hi  = np.percentile(image, high_pct)
    img = np.clip(image, lo, hi)
    img = (img - lo) / (hi - lo + 1e-9) * 255.0
    return img.astype(np.uint8)


def _apply_augmentation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply tiny geometric + photometric jitter to a uint8 image.
    - Random rotation (any angle)
    - Tiny random crop + resize back
    - Tiny intensity scale
    """
    from scipy.ndimage import rotate, zoom

    # Rotation
    angle   = rng.uniform(*AUGMENT_P.rotation_range)
    image   = rotate(image, angle, reshape=False, order=1, cval=0)

    # Tiny crop — avoids hard borders after rotation
    crop_f  = rng.uniform(*AUGMENT_P.crop_fraction_range)
    if crop_f > 0:
        h, w = image.shape
        c    = int(h * crop_f)
        if c > 0:
            image = image[c:h-c, c:w-c]
            # zoom back to original size
            image = zoom(image, h / image.shape[0], order=1)
            image = image[:h, :w]  # safety clip

    # Intensity jitter (applied before uint8 clamp at save time)
    scale   = rng.uniform(*AUGMENT_P.intensity_jitter)
    image   = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    return image


# ─────────────────────────────────────────────────────────────────────────────
# MAIN NOISE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_noise_and_realism(
    clean_image    : np.ndarray,
    realism_level  : str,
    rng            : np.random.Generator,
    fwhm_arcsec    : float,
    pixel_scale    : float = PIXEL_SCALE,
) -> Tuple[np.ndarray, Dict]:
    """
    Apply noise + realism effects to a clean ray-traced image.

    Args:
        clean_image:    Float32 array in counts (from lensing_engine).
        realism_level:  'clean' | 'semi_messy' | 'messy'
        rng:            Seeded numpy Generator.
        fwhm_arcsec:    PSF FWHM already used (for star generation).
        pixel_scale:    Arcsec/pixel.

    Returns:
        (uint8 image [0–255], noise_meta dict)
    """
    p          = NOISE_PARAMS[realism_level]
    img        = clean_image.copy().astype(np.float64)
    img_size   = img.shape[0]
    noise_meta = {"realism": realism_level}

    # ── 1. Scale to exposure time ──────────────────────────────────────────
    img *= p.exposure_time
    noise_meta["exposure_time"] = p.exposure_time

    # ── 2. Sky background ──────────────────────────────────────────────────
    if realism_level == "clean":
        sky = np.full_like(img, p.sky_background)
    elif realism_level == "semi_messy":
        sky = _sky_gradient(img_size, rng,
                            p.sky_gradient_amp, p.sky_background)
    else:  # messy
        sky = _sky_gradient(img_size, rng,
                            p.sky_gradient_amp, p.sky_background)

    img += sky
    noise_meta["sky_background"] = p.sky_background

    # ── 3. Poisson noise ──────────────────────────────────────────────────
    img_positive = np.clip(img, 0, None)
    img          = rng.poisson(img_positive).astype(np.float64)

    # ── 4. Read noise ──────────────────────────────────────────────────────
    read_noise   = rng.normal(0, p.read_noise_sigma, size=img.shape)
    img         += read_noise
    noise_meta["read_noise_sigma"] = p.read_noise_sigma

    # ── 5. Background galaxies / stars ────────────────────────────────────
    if realism_level == "semi_messy":
        n_extra  = rng.integers(*p.n_extra_range, endpoint=True)
        for _ in range(n_extra):
            _add_bg_galaxy(img, rng, p.extra_amp_range, pixel_scale)
        noise_meta["n_extra_objects"] = int(n_extra)

    elif realism_level == "messy":
        n_bg = rng.integers(*p.n_bg_galaxies_range, endpoint=True)
        for _ in range(n_bg):
            _add_bg_galaxy(img, rng, p.bg_amp_range, pixel_scale)

        n_st = rng.integers(*p.n_stars_range, endpoint=True)
        for _ in range(n_st):
            _add_star(img, rng, p.star_amp_range, fwhm_arcsec, pixel_scale)

        noise_meta["n_bg_galaxies"] = int(n_bg)
        noise_meta["n_stars"]       = int(n_st)

        # ── 6. Cosmic rays ─────────────────────────────────────────────────
        _cosmic_rays(img, rng, p.cr_fraction)
        noise_meta["cr_fraction"] = p.cr_fraction

        # ── 7. Vignetting ──────────────────────────────────────────────────
        _vignette(img, p.vignette_strength)
        noise_meta["vignette_strength"] = p.vignette_strength

    # ── 8. Photometric jitter ─────────────────────────────────────────────
    flux_scale   = rng.uniform(*p.flux_jitter)
    img         *= flux_scale
    noise_meta["flux_jitter"] = round(flux_scale, 4)

    # ── 9. Normalise to uint8 ─────────────────────────────────────────────
    img_u8 = _normalise_to_uint8(img.astype(np.float32))

    # ── 10. Augmentation (rotation + tiny crop) ───────────────────────────
    img_u8 = _apply_augmentation(img_u8, rng)

    return img_u8, noise_meta
