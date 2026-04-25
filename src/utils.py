"""
utils.py
--------
Visualization utilities for AE and VAE analysis.

Functions:
    plot_loss_curves          -- Train/val loss over epochs
    plot_vae_loss_curves      -- Train/val + recon/KL loss for VAE
    plot_reconstructions      -- Original vs reconstructed image grid
    plot_latent_space         -- 2D latent space scatter plot
    plot_generated_samples    -- VAE sample generation grid
    plot_denoising            -- Noisy input vs denoised reconstruction
    save_all_visualizations   -- Run all plots for all regions
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.model import Autoencoder, VAE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGURES_DIR: str = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Loss curves
# ---------------------------------------------------------------------------


def plot_loss_curves(
    history: Dict[str, List[float]],
    region: str,
    model_type: str,
    save: bool = True,
) -> None:
    """
    Plot training and validation loss curves over epochs.

    Args:
        history:    Loss history dict with 'train_loss' and 'val_loss'.
        region:     Anatomical region name (e.g. 'AbdomenCT').
        model_type: Either 'AE' or 'VAE'.
        save:       Whether to save the figure to disk.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    plt.title(f"{model_type} Loss — {region}", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_{model_type}_loss.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


def plot_vae_loss_curves(
    history: Dict[str, List[float]],
    region: str,
    save: bool = True,
) -> None:
    """
    Plot VAE loss curves: total, reconstruction, and KL divergence.

    Having all three on one figure clearly shows the trade-off
    between reconstruction quality and latent space regularization.

    Args:
        history: VAE loss history with keys 'train_loss', 'val_loss',
                 'train_recon_loss', 'train_kl_loss'.
        region:  Anatomical region name.
        save:    Whether to save the figure to disk.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2, linestyle="--")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Reconstruction loss
    axes[1].plot(epochs, history["train_recon_loss"], color="green", linewidth=2)
    axes[1].set_title("Reconstruction Loss")
    axes[1].set_xlabel("Epoch")

    # KL divergence
    axes[2].plot(epochs, history["train_kl_loss"], color="orange", linewidth=2)
    axes[2].set_title("KL Divergence")
    axes[2].set_xlabel("Epoch")

    fig.suptitle(f"VAE Loss Components — {region}", fontsize=14)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_VAE_loss_components.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


# ---------------------------------------------------------------------------
# Reconstruction grid
# ---------------------------------------------------------------------------


def plot_reconstructions(
    model,
    test_dataset: tf.data.Dataset,
    region: str,
    model_type: str,
    n_images: int = 8,
    save: bool = True,
) -> None:
    """
    Plot a grid of original images vs their reconstructions.

    Top row: original images.
    Bottom row: reconstructed images.

    Args:
        model:        Trained AE or VAE model.
        test_dataset: tf.data.Dataset for the test split.
        region:       Anatomical region name.
        model_type:   Either 'AE' or 'VAE'.
        n_images:     Number of image pairs to show (default: 8).
        save:         Whether to save the figure to disk.
    """
    # Grab one batch from test set
    for batch in test_dataset.take(1):
        originals = batch[:n_images]

    # Get reconstructions
    if model_type == "VAE":
        reconstructed, _, _ = model(originals, training=False)
    else:
        reconstructed = model(originals, training=False)

    originals = originals.numpy()
    reconstructed = reconstructed.numpy()

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))

    for i in range(n_images):
        # Original
        axes[0, i].imshow(originals[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_title("Original", fontsize=10, loc="left")

        # Reconstructed
        axes[1, i].imshow(reconstructed[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_title("Reconstructed", fontsize=10, loc="left")

    fig.suptitle(f"{model_type} Reconstructions — {region}", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_{model_type}_reconstructions.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


# ---------------------------------------------------------------------------
# Latent space
# ---------------------------------------------------------------------------


def plot_latent_space(
    model,
    test_dataset: tf.data.Dataset,
    region: str,
    model_type: str,
    save: bool = True,
) -> None:
    """
    Plot the 2D latent space representation of the test set.

    Since Medical MNIST is unsupervised here, points are colored
    by density using a scatter plot. The structure of the latent
    space reveals how well the model organizes the data.

    For VAE, uses mu (the mean) as the latent coordinate — this
    is standard practice since mu is the deterministic summary
    of where each image maps to in the latent space.

    Args:
        model:        Trained AE or VAE model.
        test_dataset: tf.data.Dataset for the test split.
        region:       Anatomical region name.
        model_type:   Either 'AE' or 'VAE'.
        save:         Whether to save the figure to disk.
    """
    all_z = []

    for batch in test_dataset:
        if model_type == "VAE":
            mu, _ = model.encode(batch)
            all_z.append(mu.numpy())
        else:
            z = model.encode(batch)
            all_z.append(z.numpy())

    all_z = np.concatenate(all_z, axis=0)  # (N, 2)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        all_z[:, 0],
        all_z[:, 1],
        alpha=0.3,
        s=5,
        c=np.sqrt(all_z[:, 0] ** 2 + all_z[:, 1] ** 2),
        cmap="viridis",
    )
    plt.colorbar(label="Distance from origin")
    plt.title(f"{model_type} Latent Space — {region}", fontsize=13)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_{model_type}_latent_space.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


# ---------------------------------------------------------------------------
# Generated samples (VAE only)
# ---------------------------------------------------------------------------


def plot_generated_samples(
    vae: VAE,
    region: str,
    n_samples: int = 16,
    save: bool = True,
) -> None:
    """
    Generate and plot new images by sampling from the VAE prior N(0, I).

    This only works meaningfully for VAE because its latent space
    is regularized. Sampling from N(0, I) in an AE latent space
    produces mostly noise since AE has no such regularization.

    Args:
        vae:       Trained VAE model.
        region:    Anatomical region name.
        n_samples: Number of images to generate (default: 16).
        save:      Whether to save the figure to disk.
    """
    samples = vae.sample(n_samples).numpy()

    cols = 8
    rows = n_samples // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    fig.suptitle(f"VAE Generated Samples — {region}", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_VAE_generated_samples.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def plot_denoising(
    model,
    test_dataset: tf.data.Dataset,
    region: str,
    model_type: str,
    noise_factor: float = 0.3,
    n_images: int = 8,
    save: bool = True,
) -> None:
    """
    Demonstrate denoising capability by adding Gaussian noise to
    test images and comparing model reconstructions.

    Three rows:
        Row 1: Original clean images
        Row 2: Noisy images (input to model)
        Row 3: Model reconstruction (denoised output)

    Args:
        model:        Trained AE or VAE model.
        test_dataset: tf.data.Dataset for the test split.
        region:       Anatomical region name.
        model_type:   Either 'AE' or 'VAE'.
        noise_factor: Standard deviation of Gaussian noise (default: 0.3).
        n_images:     Number of image examples to show.
        save:         Whether to save the figure to disk.
    """
    for batch in test_dataset.take(1):
        originals = batch[:n_images]

    # Add Gaussian noise and clip to [0, 1]
    noise = tf.random.normal(shape=tf.shape(originals), stddev=noise_factor)
    noisy = tf.clip_by_value(originals + noise, 0.0, 1.0)

    # Reconstruct from noisy input
    if model_type == "VAE":
        reconstructed, _, _ = model(noisy, training=False)
    else:
        reconstructed = model(noisy, training=False)

    originals = originals.numpy()
    noisy = noisy.numpy()
    reconstructed = reconstructed.numpy()

    fig, axes = plt.subplots(3, n_images, figsize=(n_images * 2, 6))
    row_labels = ["Original", "Noisy", "Denoised"]

    for i in range(n_images):
        for row, img in enumerate([originals, noisy, reconstructed]):
            axes[row, i].imshow(img[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, i].axis("off")
            if i == 0:
                axes[row, i].set_ylabel(
                    row_labels[row], fontsize=10, rotation=0,
                    labelpad=50, va="center"
                )

    fig.suptitle(f"{model_type} Denoising — {region} (noise={noise_factor})", fontsize=13)
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, f"{region}_{model_type}_denoising.png")
        plt.savefig(path, dpi=150)
        print(f"  Saved: {path}")

    plt.close()


# ---------------------------------------------------------------------------
# Run all visualizations for all regions
# ---------------------------------------------------------------------------


def save_all_visualizations(results: Dict) -> None:
    """
    Generate and save all visualizations for every region and model type.

    Call this after train_all() completes. Expects the results dict
    returned by train_all().

    Args:
        results: Dict returned by train_all(), structured as:
            {
                region: {
                    'ae': Autoencoder,
                    'vae': VAE,
                    'ae_history': dict,
                    'vae_history': dict,
                    'test_dataset': tf.data.Dataset,
                }
            }
    """
    print("\n=== Generating Visualizations ===")

    for region, data in results.items():
        print(f"\n  [{region}]")

        ae = data["ae"]
        vae = data["vae"]
        test_ds = data["test_dataset"]

        # Loss curves
        plot_loss_curves(data["ae_history"], region, "AE")
        plot_loss_curves(data["vae_history"], region, "VAE")
        plot_vae_loss_curves(data["vae_history"], region)

        # Reconstructions
        plot_reconstructions(ae, test_ds, region, "AE")
        plot_reconstructions(vae, test_ds, region, "VAE")

        # Latent space
        plot_latent_space(ae, test_ds, region, "AE")
        plot_latent_space(vae, test_ds, region, "VAE")

        # VAE generation
        plot_generated_samples(vae, region)

        # Denoising
        plot_denoising(ae, test_ds, region, "AE")
        plot_denoising(vae, test_ds, region, "VAE")

    print("\n=== All figures saved to figures/ ===")