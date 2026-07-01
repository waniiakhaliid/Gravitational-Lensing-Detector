"""
core/lensing_engine.py
Physics engine: wraps lenstronomy to produce convergence maps, deflections,
and ray-traced images for SIE + external shear + NFW subhalo systems.
"""

import numpy as np
from typing import Dict, Optional, Tuple

# lenstronomy imports — compatible with lenstronomy >= 1.11
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
import lenstronomy.Util.util as util

from config import (
    IMG_SIZE, PIXEL_SCALE, SUPERSAMPLE,
    LENS_P, SRC_P, SUB_P, MORPH_C, PSF_P,
)


# ─────────────────────────────────────────────────────────────────────────────
# PSF BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def make_psf_kernel(fwhm_arcsec: float, pixel_scale: float = PIXEL_SCALE,
                    kernel_size: int = 21) -> np.ndarray:
    """
    Build a 2D Gaussian PSF kernel that mimics HST ACS/WFC.
    kernel_size must be odd.
    Returns a normalised 2D numpy array.
    """
    sigma_pix = (fwhm_arcsec / pixel_scale) / (2 * np.sqrt(2 * np.log(2)))
    half = kernel_size // 2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma_pix**2))
    return kernel / kernel.sum()


def make_psf_object(fwhm_arcsec: float, pixel_scale: float = PIXEL_SCALE) -> PSF:
    """Return a lenstronomy PSF object (pixel-based Gaussian kernel)."""
    kernel = make_psf_kernel(fwhm_arcsec, pixel_scale, kernel_size=21)
    psf_dict = {
        "psf_type"        : "PIXEL",
        "kernel_point_source": kernel,
    }
    return PSF(**psf_dict)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DATA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def make_image_data(img_size: int = IMG_SIZE,
                    pixel_scale: float = PIXEL_SCALE) -> ImageData:
    """Build a lenstronomy ImageData object for a square cutout.
    Compatible with lenstronomy >= 1.11 (num_pix/delta_pix API).
    """
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = (
        util.make_grid_with_coordtransform(
            num_pix=img_size,
            delta_pix=pixel_scale,
            inverse=True,
        )
    )
    # nx/ny removed in lenstronomy >= 1.12 — size is inferred from image_data
    kwargs_data = {
        "ra_at_xy_0"         : ra_at_xy_0,
        "dec_at_xy_0"        : dec_at_xy_0,
        "transform_pix2angle": Mpix2coord,
        "image_data"         : np.zeros((img_size, img_size)),
    }
    return ImageData(**kwargs_data)


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SAMPLERS  (deterministic given an rng)
# ─────────────────────────────────────────────────────────────────────────────

def sample_lens_kwargs(rng: np.random.Generator,
                       morph: str = "arc") -> Tuple[Dict, Dict, Dict]:
    """
    Sample lens model kwargs for:
      - SIE (main deflector)
      - SHEAR (external)
    Also returns the sampled parameter dict for metadata logging.

    Returns:
        kwargs_lens     : list of dicts for lenstronomy LensModel
        kwargs_lens_light : list of dicts for LightModel (lens galaxy)
        meta            : flat dict of all sampled values (for CSV)
    """
    # Einstein radius
    theta_E = rng.uniform(*LENS_P.theta_E_range)

    # Ellipticity — quad morphology needs higher ellipticity
    e_scale = 2.0 if morph == "quad" else 1.0
    e1 = rng.uniform(*LENS_P.ellip_range) * e_scale
    e2 = rng.uniform(*LENS_P.ellip_range) * e_scale
    e1 = np.clip(e1, -0.45, 0.45)
    e2 = np.clip(e2, -0.45, 0.45)

    # External shear
    gamma1 = rng.uniform(*LENS_P.gamma_ext_range) * rng.choice([-1, 1])
    gamma2 = rng.uniform(*LENS_P.gamma_ext_range) * rng.choice([-1, 1])

    # Lens center
    cx = rng.uniform(*LENS_P.center_offset_range)
    cy = rng.uniform(*LENS_P.center_offset_range)

    kwargs_lens = [
        {   # SIE
            "theta_E" : theta_E,
            "e1"      : e1,
            "e2"      : e2,
            "center_x": cx,
            "center_y": cy,
        },
        {   # SHEAR
            "gamma1"  : gamma1,
            "gamma2"  : gamma2,
            "ra_0"    : 0.0,
            "dec_0"   : 0.0,
        },
    ]

    # Lens galaxy light
    R_lens  = rng.uniform(*LENS_P.R_sersic_range)
    n_lens  = rng.uniform(*LENS_P.n_sersic_lens_range)
    amp_l   = rng.uniform(*LENS_P.amp_lens_range)
    e1_l    = rng.uniform(-0.3, 0.3)
    e2_l    = rng.uniform(-0.3, 0.3)

    kwargs_lens_light = [
        {
            "amp"       : amp_l,
            "R_sersic"  : R_lens,
            "n_sersic"  : n_lens,
            "e1"        : e1_l,
            "e2"        : e2_l,
            "center_x"  : cx,
            "center_y"  : cy,
        }
    ]

    meta = {
        "lens_theta_E" : round(theta_E, 4),
        "lens_e1"      : round(e1, 4),
        "lens_e2"      : round(e2, 4),
        "lens_gamma1"  : round(gamma1, 4),
        "lens_gamma2"  : round(gamma2, 4),
        "lens_cx"      : round(cx, 4),
        "lens_cy"      : round(cy, 4),
        "lens_R_sersic": round(R_lens, 4),
        "lens_n_sersic": round(n_lens, 4),
        "lens_amp"     : round(amp_l, 2),
    }
    return kwargs_lens, kwargs_lens_light, meta


def sample_source_kwargs(rng: np.random.Generator,
                         morph: str,
                         theta_E: float) -> Tuple[Dict, Dict]:
    """
    Sample source position constrained by morphology type.
    Returns (kwargs_source_light, meta).
    """
    # Source position
    constraint = getattr(MORPH_C, morph)   # (min_frac, max_frac) of theta_E
    r_frac = rng.uniform(*constraint)
    r_src  = r_frac * theta_E
    phi    = rng.uniform(0, 2 * np.pi)
    bx     = r_src * np.cos(phi)
    by     = r_src * np.sin(phi)

    R_src   = rng.uniform(*SRC_P.R_sersic_range)
    n_src   = rng.uniform(*SRC_P.n_sersic_src_range)
    amp_s   = rng.uniform(*SRC_P.amp_src_range)
    e1_s    = rng.uniform(*SRC_P.ellip_src_range)
    e2_s    = rng.uniform(*SRC_P.ellip_src_range)

    kwargs_source = [
        {
            "amp"      : amp_s,
            "R_sersic" : R_src,
            "n_sersic" : n_src,
            "e1"       : e1_s,
            "e2"       : e2_s,
            "center_x" : bx,
            "center_y" : by,
        }
    ]
    meta = {
        "src_bx"      : round(bx, 4),
        "src_by"      : round(by, 4),
        "src_R_sersic": round(R_src, 4),
        "src_n_sersic": round(n_src, 4),
        "src_amp"     : round(amp_s, 2),
        "src_e1"      : round(e1_s, 4),
        "src_e2"      : round(e2_s, 4),
    }
    return kwargs_source, meta


def sample_subhalo_kwargs(rng: np.random.Generator,
                          theta_E: float) -> Tuple[Dict, Dict]:
    """
    Sample an NFW subhalo close to the lensed arc.
    Position is chosen to be near the Einstein ring radius.
    Returns (kwargs_subhalo_list_to_add, meta).
    """
    mass         = rng.uniform(*SUB_P.mass_range)
    concentration= rng.uniform(*SUB_P.concentration_range)

    # Place subhalo near the arc (within 0.3–1.0 × theta_E)
    r   = rng.uniform(0.3 * theta_E, 1.0 * theta_E)
    phi = rng.uniform(0, 2 * np.pi)
    sx  = r * np.cos(phi)
    sy  = r * np.sin(phi)

    # NFW: lenstronomy uses alpha_Rs and Rs
    # Convert from mass + concentration using approximate analytic relation
    # Rs [arcsec] ≈ (M_200 / (4π * rho_s * Rs^3))^(1/3) — simplified

    # Rs controls how spread out the halo is
    # small subhalo mass → Rs close to 0.05 arcsec
    # large subhalo mass → Rs close to 0.20 arcsec
    # It is not the visible size of the subhalo. It is the scale of its gravitational effect.

    # We use a phenomenological mapping that keeps subhalo SUBTLE
    # Rs [arcsec] roughly 0.05–0.2 for M ~ 1e8–5e9 M_sun
    log_mass_norm = (np.log10(mass) - 8.0) / (np.log10(5e9) - 8.0)
    Rs  = 0.05 + log_mass_norm * 0.15   # [arcsec]  0.05–0.20
    # alpha_Rs: deflection at Rs — kept very small for subtlety
    alpha_Rs = 0.01 + log_mass_norm * 0.04   # [arcsec]  0.01–0.05

    kwargs_subhalo = {
        "Rs"      : Rs,
        "alpha_Rs": alpha_Rs,
        "center_x": sx,
        "center_y": sy,
    }
    meta = {
        "subhalo_mass"    : f"{mass:.3e}",
        "subhalo_conc"    : round(concentration, 2),
        "subhalo_Rs"      : round(Rs, 4),
        "subhalo_alpha_Rs": round(alpha_Rs, 4),
        "subhalo_cx"      : round(sx, 4),
        "subhalo_cy"      : round(sy, 4),
    }
    return kwargs_subhalo, meta


# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS PARAMETER SAMPLER  (decoupled from rendering)
# ─────────────────────────────────────────────────────────────────────────────

def sample_physics_params(
    rng         : np.random.Generator,
    morph       : str,
    with_subhalo: bool,
) -> Dict:
    """
    Sample all lens + source (+ optional subhalo) parameters.
    Returns a single dict that can be passed directly to
    render_source_only_image() and render_lensed_image_from_params().

    This decoupling lets the pipeline:
      1. Sample params once.
      2. Render source-only for morphology analysis.
      3. Render the full image from the *same* params (no re-sampling).

    Keys in the returned dict
    -------------------------
    lens_model_list   : list[str]   — e.g. ["SIE", "SHEAR"] or [..., "NFW"]
    kwargs_lens       : list[dict]  — lenstronomy lens kwargs
    kwargs_lens_light : list[dict]  — lenstronomy lens-light kwargs
    kwargs_source     : list[dict]  — lenstronomy source kwargs
    theta_E           : float       — SIE Einstein radius [arcsec]
    meta_lens         : dict        — flat CSV-ready lens metadata
    meta_src          : dict        — flat CSV-ready source metadata
    meta_sub          : dict        — flat CSV-ready subhalo metadata (empty if no subhalo)
    """
    lens_model_list = ["SIE", "SHEAR"]
    if with_subhalo:
        lens_model_list.append("NFW")

    kwargs_lens, kwargs_lens_light, meta_lens = sample_lens_kwargs(rng, morph)
    theta_E = kwargs_lens[0]["theta_E"]

    kwargs_source, meta_src = sample_source_kwargs(rng, morph, theta_E)

    meta_sub: Dict = {}
    if with_subhalo:
        kwargs_sub, meta_sub = sample_subhalo_kwargs(rng, theta_E)
        kwargs_lens.append(kwargs_sub)

    return {
        "lens_model_list"  : lens_model_list,
        "kwargs_lens"      : kwargs_lens,
        "kwargs_lens_light": kwargs_lens_light,
        "kwargs_source"    : kwargs_source,
        "theta_E"          : theta_E,
        "meta_lens"        : meta_lens,
        "meta_src"         : meta_src,
        "meta_sub"         : meta_sub,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-ONLY RENDERER  (no PSF · no lens light — for morphology analysis)
# ─────────────────────────────────────────────────────────────────────────────

def render_source_only_image(
    params     : Dict,
    img_size   : int   = IMG_SIZE,
    pixel_scale: float = PIXEL_SCALE,
) -> np.ndarray:
    """
    Render the lensed arc with **no PSF** and **no lens light**.

    This is the image passed to analyze_morphology() and must never have
    PSF, noise, or lens-galaxy light applied — doing so would corrupt the
    morphology measurement.

    Parameters
    ----------
    params : dict
        Output of sample_physics_params().

    Returns
    -------
    np.ndarray (float32, H × W)
        Raw lenstronomy source counts — physically meaningful but
        instrument-free.
    """
    # PSF type "NONE" → lenstronomy skips convolution entirely
    psf_obj  = PSF(psf_type="NONE")
    data_obj = make_image_data(img_size, pixel_scale)

    lens_model_obj = LensModel(lens_model_list=params["lens_model_list"])
    source_model   = LightModel(light_model_list=["SERSIC_ELLIPSE"])

    kwargs_numerics = {
        "supersampling_factor"     : SUPERSAMPLE,
        "supersampling_convolution": False,
    }

    image_model = ImageModel(
        data_class             = data_obj,
        psf_class              = psf_obj,
        lens_model_class       = lens_model_obj,
        source_model_class     = source_model,
        lens_light_model_class = None,       # no lens galaxy light
        kwargs_numerics        = kwargs_numerics,
    )

    image = image_model.image(
        kwargs_lens       = params["kwargs_lens"],
        kwargs_source     = params["kwargs_source"],
        kwargs_lens_light = [],
        kwargs_ps         = None,
    )
    return image.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# FULL RENDERER  (PSF + lens light — from pre-sampled params)
# ─────────────────────────────────────────────────────────────────────────────

def render_lensed_image_from_params(
    params     : Dict,
    fwhm_arcsec: float,
    img_size   : int   = IMG_SIZE,
    pixel_scale: float = PIXEL_SCALE,
) -> Tuple[np.ndarray, Dict]:
    """
    Render the complete lensed image (source arc + lens galaxy light, with PSF)
    from pre-sampled physics parameters.

    Use this after analyze_morphology() has approved the arc morphology so that
    the same physical realisation is used for both the analysis and final image.

    Parameters
    ----------
    params : dict
        Output of sample_physics_params().
    fwhm_arcsec : float
        PSF FWHM in arcseconds (sampled in image_generator before the retry loop).

    Returns
    -------
    (image float32 H×W, metadata dict)
    """
    psf_obj  = make_psf_object(fwhm_arcsec, pixel_scale)
    data_obj = make_image_data(img_size, pixel_scale)

    lens_model_obj   = LensModel(lens_model_list=params["lens_model_list"])
    source_model     = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    lens_light_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])

    kwargs_numerics = {
        "supersampling_factor"     : SUPERSAMPLE,
        "supersampling_convolution": False,
    }

    image_model = ImageModel(
        data_class              = data_obj,
        psf_class               = psf_obj,
        lens_model_class        = lens_model_obj,
        source_model_class      = source_model,
        lens_light_model_class  = lens_light_model,
        kwargs_numerics         = kwargs_numerics,
    )

    image = image_model.image(
        kwargs_lens       = params["kwargs_lens"],
        kwargs_source     = params["kwargs_source"],
        kwargs_lens_light = params["kwargs_lens_light"],
        kwargs_ps         = None,
    )

    meta = {
        "fwhm_arcsec": round(fwhm_arcsec, 4),
        **params["meta_lens"],
        **params["meta_src"],
        **params["meta_sub"],
    }
    return image.astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPATIBLE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def render_lensed_image(rng: np.random.Generator,
                        morph: str,
                        with_subhalo: bool,
                        fwhm_arcsec: float,
                        img_size: int = IMG_SIZE,
                        pixel_scale: float = PIXEL_SCALE) -> Tuple[np.ndarray, Dict]:
    """
    Original one-shot API: sample parameters, then render the full lensed image.
    Preserved for backward compatibility with any code that calls it directly.

    For the morphology-aware pipeline use:
        params = sample_physics_params(rng, morph, with_subhalo)
        src    = render_source_only_image(params)
        ...analyze_morphology(src, ...)...
        img, meta = render_lensed_image_from_params(params, fwhm_arcsec)
    """
    params = sample_physics_params(rng, morph, with_subhalo)
    return render_lensed_image_from_params(params, fwhm_arcsec, img_size, pixel_scale)

def render_no_lens_image(rng: np.random.Generator,
                         fwhm_arcsec: float,
                         img_size: int = IMG_SIZE,
                         pixel_scale: float = PIXEL_SCALE) -> Tuple[np.ndarray, Dict]:

    """
    Render a field of non-lensed galaxies.
    Returns (image_array [float32, counts], metadata_dict).
    """
    from config import NO_LENS_P

    psf_obj  = make_psf_object(fwhm_arcsec, pixel_scale)
    data_obj = make_image_data(img_size, pixel_scale)

    n_gal = rng.integers(*NO_LENS_P.n_galaxies_range, endpoint=True)
    half  = (img_size * pixel_scale) / 2.0

    light_list  = ["SERSIC_ELLIPSE"] * n_gal
    light_model = LightModel(light_model_list=light_list)

    kwargs_light = []
    for _ in range(n_gal):
        amp = rng.uniform(*NO_LENS_P.amp_range)
        R   = rng.uniform(*NO_LENS_P.R_sersic_range)
        n   = rng.uniform(*NO_LENS_P.n_sersic_range)
        e1  = rng.uniform(*NO_LENS_P.ellip_range)
        e2  = rng.uniform(*NO_LENS_P.ellip_range)
        cx  = rng.uniform(-half * 0.6, half * 0.6)
        cy  = rng.uniform(-half * 0.6, half * 0.6)
        kwargs_light.append({
            "amp": amp, "R_sersic": R, "n_sersic": n,
            "e1": e1, "e2": e2, "center_x": cx, "center_y": cy,
        })

    # Use a dummy lens model (no deflection)
    lens_model_obj = LensModel(lens_model_list=[])

    kwargs_numerics = {"supersampling_factor": SUPERSAMPLE,
                       "supersampling_convolution": False}

    image_model = ImageModel(
        data_class             = data_obj,
        psf_class              = psf_obj,
        lens_model_class       = lens_model_obj,
        source_model_class     = light_model,
        lens_light_model_class = None,
        kwargs_numerics        = kwargs_numerics,
    )

    image = image_model.image(
        kwargs_lens       = [],
        kwargs_source     = kwargs_light,
        kwargs_lens_light = [],
        kwargs_ps         = None,
    )

    meta = {
        "fwhm_arcsec" : round(fwhm_arcsec, 4),
        "n_galaxies"  : n_gal,
    }
    return image.astype(np.float32), meta