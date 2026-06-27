# Gravitational Lensing Substructure Detection

This project focuses on generating HST-like simulated gravitational lensing images for dark matter substructure detection.

The first part of the project is a data generation pipeline. It creates simulated lensing images with realistic telescope effects such as PSF, noise, background variation, and different lens morphologies.

Later parts of the project will include model training, evaluation, and deployment.

## Project Goals

* Generate realistic HST-like gravitational lensing images
* Create labeled datasets for machine learning
* Train models to classify lensing images
* Detect possible dark matter substructure
* Evaluate model performance using test images and visual results

## Current Project Structure

```text
gravitational-lensing-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data_generation/
│   └── hst_pipeline/
│       ├── config.py
│       ├── core/
│       ├── generators/
│       ├── utils/
│       └── notebooks/
├── samples/
├── results/
└── models/
```

## Current Module: Data Generation

The `hst_pipeline` module is responsible for generating simulated HST-like lensing images.

It includes:

* lensing physics using lenstronomy
* PSF and noise simulation
* single image generation
* batch dataset generation
* metadata CSV creation
* visualization and quality-check tools

## Dataset Classes

The planned dataset may include:

* no lens
* lens without subhalo
* lens with subhalo

Additional morphology labels may include:

* ring
* arc
* double
* quad
* partial ring

## Large Files

Large datasets, trained models, and generated image folders are not stored on GitHub.

They should be stored in:

* Google Drive
* Kaggle Dataset
* Hugging Face Dataset

GitHub is used only for code, documentation, small samples, and result plots.

## Important Note

This project is built for learning and experimentation. The model will be trained mainly on simulated images. Predictions on real telescope images should be treated as candidate results only and require proper astrophysical lens modeling for scientific confirmation.

## Tools Used

* Python
* NumPy
* Matplotlib
* Pandas
* PyTorch
* Lenstronomy
* Astropy
* Google Colab
* Google Drive
* GitHub
