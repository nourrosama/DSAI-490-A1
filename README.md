# DSAI 490 — Assignment 1: Autoencoders (AE & VAE)

## Overview
Implementation of Autoencoder (AE) and Variational Autoencoder (VAE)
for representation learning on the Medical MNIST dataset.
A separate AE and VAE is trained for each of the 6 anatomical regions.

## Dataset
- Source: https://www.kaggle.com/datasets/andrewmvd/medical-mnist
- 6 anatomical regions: AbdomenCT, BreastMRI, ChestCT, CXR, Hand, HeadCT
- Images: 64×64 grayscale JPEG

**Dataset is not included in this repo.** To set up:
1. Download from Kaggle link above
2. Extract into `data/raw/` so each region has its own subfolder

## Project Structure
├── data/
│   ├── raw/          ← dataset lives here (not tracked by git)
│   └── processed/
├── figures/          ← generated plots (not tracked by git)
├── models/           ← saved model weights (not tracked by git)
├── notebooks/
│   └── experiment.ipynb
├── src/
│   ├── init.py
│   ├── data_processing.py   ← tf.data pipeline
│   ├── model.py             ← AE and VAE architectures
│   ├── train.py             ← training loops
│   └── utils.py             ← visualization utilities
├── tests/
├── run_experiment.py        ← master script
├── README.md
└── requirements.txt

## Setup
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Run
```bash
python run_experiment.py
```

## Architecture
### AE
- Encoder: Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(2)
- Decoder: Dense(8192) → Reshape → ConvTranspose × 3 → Sigmoid
- Latent dim: 2 (for direct 2D visualization)
- Loss: MSE

### VAE
- Encoder: Same conv stack → Dense(256) → [mu head, log_var head]
- Reparameterization: z = mu + eps * exp(0.5 * log_var)
- Decoder: Same as AE decoder
- Loss: MSE + KL divergence

## Results
