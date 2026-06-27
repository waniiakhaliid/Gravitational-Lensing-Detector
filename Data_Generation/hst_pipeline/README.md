# HST Gravitational Lensing Dataset Pipeline

Synthetic **HST-like** dataset generator for CNN-based gravitational lens detection, morphology classification, and dark matter subhalo detection.

This pipeline is the **data generation part** of the larger gravitational lensing project.

---

## Project Status

This pipeline is currently under development.

Dataset counts, physics ranges, realism levels, and output formats may change as the project improves.

---

## Dataset Summary

| Class               |      Images | Notes                                       |
| ------------------- | ----------: | ------------------------------------------- |
| `no_lens`           |      50,000 | Unlensed galaxy fields                      |
| `lens_no_subhalo`   |      25,000 | Lensed arcs / rings, no substructure        |
| `lens_with_subhalo` |      25,000 | Lensed arcs / rings with subtle NFW subhalo |
| **Total**           | **100,000** | Full planned dataset                        |

---

## Realism Levels

| Realism      | Fraction | Images | What it adds                                           |
| ------------ | -------: | -----: | ------------------------------------------------------ |
| `clean`      |      20% | 20,000 | Lens/source only, minimal noise                        |
| `semi_messy` |      30% | 30,000 | PSF + moderate noise + 1–2 faint objects               |
| `messy`      |      50% | 50,000 | HST-like noise, background galaxies, stars, vignetting |

The goal is to train the model on both simple and realistic images.

Clean images help the model learn the basic lensing pattern. Messy images help the model become more robust for real telescope-like data.

---

## Morphology Labels

Morphology labels are used only for lensed images.

| Morphology     | Fraction of lensed images | Description                       |
| -------------- | ------------------------: | --------------------------------- |
| `ring`         |                       20% | Near-complete Einstein ring       |
| `arc`          |                       20% | Single elongated arc              |
| `double`       |                       20% | Two visible images of the source  |
| `quad`         |                       20% | Four visible images of the source |
| `partial_ring` |                       20% | Around 50–70% Einstein ring       |

---

## Model Labels

| Model                                   | Label column    | Classes                                         | Images used       |
| --------------------------------------- | --------------- | ----------------------------------------------- | ----------------- |
| **Model 1** — lens detection            | `lens_label`    | `lens` / `no_lens`                              | All 100k          |
| **Model 2** — morphology classification | `lens_type`     | `ring`, `arc`, `double`, `quad`, `partial_ring` | 50k lensed images |
| **Model 3** — subhalo detection         | `subhalo_label` | `subhalo` / `no_subhalo`                        | 50k lensed images |

---

## Training Stages

| Stage       | Realism used                     | Purpose                                                     |
| ----------- | -------------------------------- | ----------------------------------------------------------- |
| **Stage 1** | `clean` only                     | Learn the core lensing signal without heavy noise           |
| **Stage 2** | `clean` + `semi_messy` + `messy` | Improve robustness to noise, stars, and background objects  |
| **Stage 3** | Real HST images                  | External testing / visual comparison on real telescope data |

Stage 3 should be treated as experimental testing, not scientific confirmation.

---

## Train / Validation / Test Split

Default split:

| Split      | Fraction |
| ---------- | -------: |
| Train      |      70% |
| Validation |      15% |
| Test       |      15% |

The split is stored in the metadata using the `split` column.

---

## File Structure

```text
hst_pipeline/
├── config.py                      # All main settings: counts, physics ranges, noise
├── core/
│   ├── lensing_engine.py          # lenstronomy physics simulation
│   └── noise_engine.py            # PSF, Poisson noise, read noise, realism layers
├── generators/
│   ├── image_generator.py         # Single-image generation orchestrator
│   └── batch_runner.py            # Batch generation + metadata CSV
├── utils/
│   ├── dataset_loader.py          # PyTorch Dataset + DataLoader factory
│   └── visualize.py               # QC plots and sample grids
└── notebooks/
    └── HST_Lensing_Dataset_Generator.ipynb  # Full Colab walkthrough
```

---

## Output Structure

The generated dataset should be saved outside GitHub, usually in Google Drive.

Example output structure:

```text
lensing_dataset/
├── images/
│   ├── no_lens/
│   ├── lens_no_subhalo/
│   └── lens_with_subhalo/
├── metadata.csv
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Large generated datasets should not be uploaded directly to GitHub.

---

## Quick Start: Google Colab

### 1. Install Dependencies

```python
!pip install lenstronomy==1.11.8 pillow tqdm scipy numpy pandas matplotlib -q
```

### 2. Clone the Repository in Colab

```python
!git clone https://github.com/YOUR_USERNAME/gravitational-lensing-project.git
%cd gravitational-lensing-project
```

Then add the pipeline folder to the Python path:

```python
import sys
sys.path.insert(0, '/content/gravitational-lensing-project/data_generation/hst_pipeline')
```

This allows Colab to import files from the `hst_pipeline` module.


### 3. Generate Dataset

```python
from generators.batch_runner import run_pipeline

run_pipeline(num_workers=1, resume=True)
```

### 4. Load Dataset for Training

```python
from utils.dataset_loader import make_dataloaders

# Stage 1 — Model 1: binary lens detection using clean images only
loaders = make_dataloaders(model_name='model1', stage=1, batch_size=64)

# Stage 2 — Model 3: subhalo detection using all realism levels
loaders = make_dataloaders(model_name='model3', stage=2, batch_size=64)
```

---

## Physics Parameters

Key default parameters:

| Parameter           | Range                  | Notes                            |
| ------------------- | ---------------------- | -------------------------------- |
| Einstein radius θ_E | 0.5–2.0 arcsec         | SIE lens profile                 |
| PSF FWHM            | 0.08–0.13 arcsec       | ACS/WFC F814W-like approximation |
| Pixel scale         | 0.05 arcsec/pixel      | ACS/WFC-like scale               |
| Image size          | 128×128 px             | Around 6.4×6.4 arcsec field      |
| Subhalo mass        | 10⁸–5×10⁹ M☉           | NFW subhalo, kept subtle         |
| Source β            | Morphology-constrained | Helps control arc/ring type      |

All parameters are controlled from `config.py`.

---

## Estimated Generation Time

Generation time depends on Colab CPU availability, image size, number of workers, and lenstronomy settings.

The values below are rough estimates and should be benchmarked again after code changes.

| Workers | Rough time for 100k images |
| ------: | -------------------------: |
|       1 |                  5–8 hours |
|       4 |              1.5–2.5 hours |

`lenstronomy` generation is mostly CPU-bound, so Colab GPU is not heavily used during image generation.

Tip: use `resume=True`. If Colab disconnects, the pipeline can continue from the previous progress.

---

## Output Metadata Columns

The pipeline saves metadata for every generated image.

```text
filename, idx, seed,
main_class, lens_label, lens_type, subhalo_label, realism, split,
fwhm_arcsec,
lens_theta_E, lens_e1, lens_e2, lens_gamma1, lens_gamma2,
lens_cx, lens_cy, lens_R_sersic, lens_n_sersic, lens_amp,
src_bx, src_by, src_R_sersic, src_n_sersic, src_amp, src_e1, src_e2,
subhalo_mass, subhalo_conc, subhalo_Rs, subhalo_alpha_Rs, subhalo_cx, subhalo_cy,
exposure_time, sky_background, read_noise_sigma, flux_jitter,
n_bg_galaxies, n_stars, cr_fraction, vignette_strength, n_extra_objects
```

This metadata can be used for training, filtering, debugging, and scientific inspection of the simulated dataset.

---



## Limitations

* Images are simulated, not real telescope observations.
* HST-like realism is approximate.
* Subhalo labels are based on simulation parameters, not confirmed real dark matter detections.
* Real HST testing should be treated as external testing only.
* A CNN prediction alone is not scientific proof of dark matter substructure.
* Scientific confirmation requires proper lens modeling and expert validation.

---

## Purpose

This pipeline is designed for learning, experimentation, and building a machine learning workflow around gravitational lensing.

It is not intended to replace professional astrophysical lens modeling.
