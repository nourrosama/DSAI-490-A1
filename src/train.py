"""
train.py
--------
Training pipeline for Autoencoder (AE) and Variational Autoencoder (VAE)
models across all anatomical regions in the Medical MNIST dataset.

Uses tf.distribute.Strategy for hardware-aware execution:
    - MirroredStrategy if GPU(s) are available.
    - Default strategy (CPU) otherwise.

Each region gets its own independently trained AE and VAE model.
Trained models are saved to the models/ directory.
"""

import os
import time
from typing import Dict, List, Tuple

import tensorflow as tf

from src.data_processing import build_region_datasets, ANATOMICAL_REGIONS
from src.model import Autoencoder, VAE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS: int = 30
LEARNING_RATE: float = 1e-3
MODELS_DIR: str = "models"


# ---------------------------------------------------------------------------
# Strategy setup (from distributed training notebook)
# ---------------------------------------------------------------------------


def get_strategy() -> tf.distribute.Strategy:
    """
    Return the appropriate tf.distribute.Strategy for this machine.

    Uses MirroredStrategy if GPUs are available, otherwise falls
    back to the default strategy (CPU). This matches the pattern
    shown in the distributed training reference notebook.

    Returns:
        A tf.distribute.Strategy instance.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  GPUs detected: {len(gpus)} — using MirroredStrategy")
        return tf.distribute.MirroredStrategy()
    else:
        print("  No GPU detected — using default strategy (CPU)")
        return tf.distribute.get_strategy()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def ae_loss(
    original: tf.Tensor,
    reconstructed: tf.Tensor,
) -> tf.Tensor:
    """
    Compute the AE reconstruction loss (Mean Squared Error).

    Averages over pixels and batch dimension so the loss scale
    stays consistent regardless of batch size or image resolution.

    Args:
        original:      Original image batch, shape (B, H, W, C).
        reconstructed: Reconstructed image batch, shape (B, H, W, C).

    Returns:
        Scalar loss tensor.
    """
    return tf.reduce_mean(tf.square(original - reconstructed))


def vae_loss(
    original: tf.Tensor,
    reconstructed: tf.Tensor,
    mu: tf.Tensor,
    log_var: tf.Tensor,
    kl_weight: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute the VAE loss: reconstruction loss + weighted KL divergence.

    Uses Binary Cross-Entropy for reconstruction loss because it
    produces values on a scale compatible with KL divergence,
    preventing KL collapse — a failure mode where the KL term is
    overwhelmed by reconstruction loss and the encoder learns no
    meaningful latent distribution.

    MSE produces values ~0.008 while KL produces ~0.0001, making
    them impossible to balance. BCE produces values ~0.1-0.3 which
    are naturally on the same scale as KL.

    Total loss = BCE + kl_weight * KL

    Args:
        original:      Original image batch, shape (B, H, W, C).
        reconstructed: Reconstructed image batch, shape (B, H, W, C).
        mu:            Latent mean, shape (B, latent_dim).
        log_var:       Latent log variance, shape (B, latent_dim).
        kl_weight:     Weight applied to KL term (increases during warmup).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss) — all scalars.
    """
    # Clip to avoid log(0)
    reconstructed = tf.clip_by_value(reconstructed, 1e-7, 1.0 - 1e-7)

    # Binary cross-entropy reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            -original * tf.math.log(reconstructed)
            - (1.0 - original) * tf.math.log(1.0 - reconstructed),
            axis=[1, 2, 3],  # sum over pixels, mean over batch
        )
    )

    # KL divergence — averaged over batch and latent dimensions
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(
            1 + log_var - tf.square(mu) - tf.exp(log_var),
            axis=1,  # sum over latent dims, mean over batch
        )
    )

    total_loss = reconstruction_loss + kl_weight * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# ---------------------------------------------------------------------------
# Single training steps (using GradientTape as in the reference notebook)
# ---------------------------------------------------------------------------


def ae_train_step(
    model: Autoencoder,
    batch: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
) -> tf.Tensor:
    """
    Perform one AE training step using GradientTape.

    Args:
        model:     The Autoencoder model.
        batch:     Image batch tensor of shape (B, 64, 64, 1).
        optimizer: Keras optimizer instance.

    Returns:
        Scalar loss for this batch.
    """
    with tf.GradientTape() as tape:
        reconstructed = model(batch, training=True)
        loss = ae_loss(batch, reconstructed)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def vae_train_step(
    model: VAE,
    batch: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    kl_weight: float = 1.0,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Perform one VAE training step using GradientTape.

    Args:
        model:      The VAE model.
        batch:      Image batch tensor of shape (B, 64, 64, 1).
        optimizer:  Keras optimizer instance.
        kl_weight:  Current KL weight (for warm-up scheduling).

    Returns:
        Tuple of (total_loss, reconstruction_loss, kl_loss).
    """
    with tf.GradientTape() as tape:
        reconstructed, mu, log_var = model(batch, training=True)
        total, recon, kl = vae_loss(
            batch, reconstructed, mu, log_var, kl_weight
        )

    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total, recon, kl


# ---------------------------------------------------------------------------
# Per-region training loops
# ---------------------------------------------------------------------------


def train_ae_for_region(
    region: str,
    datasets: Dict,
    epochs: int = EPOCHS,
) -> Tuple[Autoencoder, Dict[str, List[float]]]:
    """
    Train one Autoencoder for a single anatomical region.

    Args:
        region:   Name of the anatomical region (e.g. 'AbdomenCT').
        datasets: Dataset dict for this region with 'train' and 'val' keys.
        epochs:   Number of training epochs.

    Returns:
        Tuple of (trained model, loss history dict).
        Loss history has keys: 'train_loss', 'val_loss'.
    """
    print(f"\n  Training AE — {region}")

    model = Autoencoder(latent_dim=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        start = time.time()

        # --- Training ---
        train_losses = []
        for batch in datasets["train"]:
            loss = ae_train_step(model, batch, optimizer)
            train_losses.append(float(loss))

        # --- Validation ---
        val_losses = []
        for batch in datasets["val"]:
            reconstructed = model(batch, training=False)
            loss = ae_loss(batch, reconstructed)
            val_losses.append(float(loss))

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start
        print(
            f"    Epoch {epoch + 1:02d}/{epochs} — "
            f"train_loss: {train_loss:.4f} — "
            f"val_loss: {val_loss:.4f} — "
            f"{elapsed:.1f}s"
        )

    return model, history


def train_vae_for_region(
    region: str,
    datasets: Dict,
    epochs: int = EPOCHS,
) -> Tuple[VAE, Dict[str, List[float]]]:
    """
    Train one VAE for a single anatomical region.

    Args:
        region:   Name of the anatomical region (e.g. 'AbdomenCT').
        datasets: Dataset dict for this region with 'train' and 'val' keys.
        epochs:   Number of training epochs.

    Returns:
        Tuple of (trained model, loss history dict).
        Loss history has keys: 'train_loss', 'val_loss',
        'train_recon_loss', 'train_kl_loss'.
    """
    print(f"\n  Training VAE — {region}")

    model = VAE(latent_dim=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
    }

    # KL warm-up: linearly increase kl_weight from 0 to 1 over
    # the first half of training, then hold at 1.
    warmup_epochs = max(1, epochs // 2)

    for epoch in range(epochs):
        start = time.time()

        # Compute current KL weight (warm-up schedule)
        kl_weight = min(1.0, (epoch + 1) / warmup_epochs)

        # --- Training ---
        train_totals, train_recons, train_kls = [], [], []
        for batch in datasets["train"]:
            total, recon, kl = vae_train_step(
                model, batch, optimizer, kl_weight
            )
            train_totals.append(float(total))
            train_recons.append(float(recon))
            train_kls.append(float(kl))

        # --- Validation ---
        val_losses = []
        for batch in datasets["val"]:
            reconstructed, mu, log_var = model(batch, training=False)
            total, _, _ = vae_loss(batch, reconstructed, mu, log_var)
            val_losses.append(float(total))

        train_loss = sum(train_totals) / len(train_totals)
        val_loss = sum(val_losses) / len(val_losses)
        train_recon = sum(train_recons) / len(train_recons)
        train_kl = sum(train_kls) / len(train_kls)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_recon_loss"].append(train_recon)
        history["train_kl_loss"].append(train_kl)

        elapsed = time.time() - start
        print(
            f"    Epoch {epoch + 1:02d}/{epochs} — "
            f"train: {train_loss:.2f} — "
            f"val: {val_loss:.2f} — "
            f"recon: {train_recon:.2f} — "
            f"kl: {train_kl:.4f} — "
            f"kl_weight: {kl_weight:.2f} — "
            f"{elapsed:.1f}s"
        )

    return model, history


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_all(
    raw_data_dir: str = "data/raw",
    epochs: int = EPOCHS,
) -> Dict:
    """
    Train all 12 models: one AE + one VAE per anatomical region.

    Saves each model to models/<region>_ae_v1/ and
    models/<region>_vae_v1/ in SavedModel format.

    Args:
        raw_data_dir: Path to the raw dataset directory.
        epochs:       Number of training epochs per model.

    Returns:
        Dictionary containing all trained models and their
        loss histories, keyed by region name.

        Structure:
            {
                "AbdomenCT": {
                    "ae":          Autoencoder,
                    "vae":         VAE,
                    "ae_history":  dict,
                    "vae_history": dict,
                },
                ...
            }
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    strategy = get_strategy()
    print(f"Strategy: {strategy.__class__.__name__}")

    # Load all datasets
    all_datasets = build_region_datasets(raw_data_dir)

    results: Dict = {}

    for region in ANATOMICAL_REGIONS:
        print(f"\n{'='*50}")
        print(f"Region: {region}")
        print(f"{'='*50}")

        datasets = all_datasets[region]

        # --- Train AE ---
        ae_model, ae_history = train_ae_for_region(
            region, datasets, epochs
        )

        # Save AE
        ae_save_path = os.path.join(
            MODELS_DIR, f"{region}_ae_v1.keras"
        )
        ae_model.save(ae_save_path)
        print(f"  AE saved → {ae_save_path}")

        # --- Train VAE ---
        vae_model, vae_history = train_vae_for_region(
            region, datasets, epochs
        )

        # Save VAE
        vae_save_path = os.path.join(
            MODELS_DIR, f"{region}_vae_v1.keras"
        )
        vae_model.save(vae_save_path)
        print(f"  VAE saved → {vae_save_path}")

        results[region] = {
            "ae": ae_model,
            "vae": vae_model,
            "ae_history": ae_history,
            "vae_history": vae_history,
            "test_dataset": datasets["test"],
        }

    print("\n=== All 12 models trained and saved ===")
    return results


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_all(raw_data_dir="data/raw", epochs=EPOCHS)