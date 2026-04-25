"""
model.py
--------
Defines the Autoencoder (AE) and Variational Autoencoder (VAE)
architectures for the Medical MNIST dataset.

Both models use a 2D latent space to allow direct visualization
of learned representations without dimensionality reduction.

Architecture:
    Encoder: Conv2D stack → Dense(latent_dim)
    Decoder: Dense → Reshape → Conv2DTranspose stack → output image

Classes:
    Encoder        -- Shared convolutional encoder backbone
    Decoder        -- Shared convolutional decoder backbone
    Autoencoder    -- Deterministic AE (Encoder + Decoder)
    VAEEncoder     -- Probabilistic encoder (outputs mu and log_var)
    VAE            -- Variational AE with reparameterization trick
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LATENT_DIM: int = 2          # 2D for direct visualization
IMAGE_SHAPE: Tuple[int, int, int] = (64, 64, 1)


# ---------------------------------------------------------------------------
# Shared Encoder Backbone
# ---------------------------------------------------------------------------


class Encoder(keras.Model):
    """
    Convolutional encoder that maps an image to a latent vector.

    Architecture:
        Conv2D(32, 3, stride=2) → ReLU   # 64x64 → 32x32
        Conv2D(64, 3, stride=2) → ReLU   # 32x32 → 16x16
        Conv2D(128, 3, stride=2) → ReLU  # 16x16 → 8x8
        Flatten
        Dense(latent_dim)

    Args:
        latent_dim: Dimensionality of the latent space (default: 2).
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initialize encoder layers."""
        super().__init__()

        self.conv1 = layers.Conv2D(
            32, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv2D(
            64, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv2D(
            128, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(latent_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x:        Input image tensor of shape (B, 64, 64, 1).
            training: Whether in training mode (affects BatchNorm/Dropout).

        Returns:
            Latent vector of shape (B, latent_dim).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.dense(x)


# ---------------------------------------------------------------------------
# Shared Decoder Backbone
# ---------------------------------------------------------------------------


class Decoder(keras.Model):
    """
    Convolutional decoder that maps a latent vector back to an image.

    Architecture:
        Dense(8*8*128) → Reshape(8, 8, 128)
        Conv2DTranspose(128, 3, stride=2) → ReLU  # 8x8  → 16x16
        Conv2DTranspose(64, 3, stride=2)  → ReLU  # 16x16 → 32x32
        Conv2DTranspose(32, 3, stride=2)  → ReLU  # 32x32 → 64x64
        Conv2DTranspose(1, 3, stride=1)   → Sigmoid  # output image

    Args:
        latent_dim: Dimensionality of the latent space (default: 2).
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initialize decoder layers."""
        super().__init__()

        # Project latent vector back to spatial feature map
        self.dense = layers.Dense(8 * 8 * 128, activation="relu")
        self.reshape = layers.Reshape((8, 8, 128))

        self.deconv1 = layers.Conv2DTranspose(
            128, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.deconv2 = layers.Conv2DTranspose(
            64, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.deconv3 = layers.Conv2DTranspose(
            32, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        # Final layer: sigmoid to keep output in [0, 1]
        self.output_layer = layers.Conv2DTranspose(
            1, kernel_size=3, strides=1, padding="same", activation="sigmoid"
        )

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass through the decoder.

        Args:
            z:        Latent vector of shape (B, latent_dim).
            training: Whether in training mode.

        Returns:
            Reconstructed image tensor of shape (B, 64, 64, 1).
        """
        x = self.dense(z)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# Autoencoder (AE)
# ---------------------------------------------------------------------------


class Autoencoder(keras.Model):
    """
    Deterministic Autoencoder combining Encoder and Decoder.

    The AE learns a compressed representation by minimizing
    reconstruction loss (MSE) between input and output images.

    Args:
        latent_dim: Dimensionality of the latent space (default: 2).

    Example:
        >>> ae = Autoencoder(latent_dim=2)
        >>> reconstructed = ae(image_batch)  # shape: (B, 64, 64, 1)
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initialize AE with encoder and decoder."""
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Full forward pass: encode then decode.

        Args:
            x:        Input image tensor of shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Reconstructed image tensor of shape (B, 64, 64, 1).
        """
        z = self.encoder(x, training=training)
        return self.decoder(z, training=training)

    def encode(self, x: tf.Tensor) -> tf.Tensor:
        """
        Encode images to latent vectors (inference only).

        Args:
            x: Input image tensor of shape (B, 64, 64, 1).

        Returns:
            Latent vectors of shape (B, latent_dim).
        """
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """
        Decode latent vectors to images (inference only).

        Args:
            z: Latent vector of shape (B, latent_dim).

        Returns:
            Reconstructed image tensor of shape (B, 64, 64, 1).
        """
        return self.decoder(z, training=False)

    def get_config(self) -> dict:
        """Return model config for serialization."""
        return {"latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config: dict) -> "Autoencoder":
        """Reconstruct model from config."""
        return cls(**config)


# ---------------------------------------------------------------------------
# VAE Encoder (outputs mu and log_var)
# ---------------------------------------------------------------------------


class VAEEncoder(keras.Model):
    """
    Probabilistic encoder for the VAE.

    Instead of outputting a single latent vector, outputs the
    parameters (mu, log_var) of a Gaussian distribution over
    the latent space.

    Architecture:
        Same Conv stack as Encoder, but final Dense layer splits
        into two heads: mu and log_var, each of size latent_dim.

    Args:
        latent_dim: Dimensionality of the latent space (default: 2).
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initialize VAE encoder layers."""
        super().__init__()

        self.conv1 = layers.Conv2D(
            32, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.conv2 = layers.Conv2D(
            64, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.conv3 = layers.Conv2D(
            128, kernel_size=3, strides=2, padding="same", activation="relu"
        )
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation="relu")

        # Two separate heads for mean and log variance
        self.mu_layer = layers.Dense(latent_dim, name="mu")
        self.log_var_layer = layers.Dense(latent_dim, name="log_var")

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass through the VAE encoder.

        Args:
            x:        Input image tensor of shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Tuple of (mu, log_var), each of shape (B, latent_dim).
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)
        return mu, log_var


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------


class VAE(keras.Model):
    """
    Variational Autoencoder with reparameterization trick.

    The VAE learns a probabilistic latent space by:
        1. Encoding inputs to (mu, log_var) via VAEEncoder.
        2. Sampling z ~ N(mu, exp(log_var)) via reparameterization.
        3. Decoding z back to image space via Decoder.

    Loss = Reconstruction Loss (MSE) + KL Divergence

    The KL divergence term regularizes the latent space to be
    close to a standard normal N(0, I), enabling generation of
    new samples by sampling from N(0, I) directly.

    Args:
        latent_dim: Dimensionality of the latent space (default: 2).

    Example:
        >>> vae = VAE(latent_dim=2)
        >>> reconstructed, mu, log_var = vae(image_batch)
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        """Initialize VAE with probabilistic encoder and decoder."""
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(
        self, mu: tf.Tensor, log_var: tf.Tensor
    ) -> tf.Tensor:
        """
        Apply the reparameterization trick to sample from N(mu, var).

        Instead of sampling z ~ N(mu, var) directly (which blocks
        gradient flow), we sample epsilon ~ N(0, I) and compute:
            z = mu + epsilon * exp(0.5 * log_var)

        This keeps the sampling operation outside the gradient path.

        Args:
            mu:      Mean of the latent distribution, shape (B, latent_dim).
            log_var: Log variance of the latent distribution, shape (B, latent_dim).

        Returns:
            Sampled latent vector z of shape (B, latent_dim).
        """
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + epsilon * tf.exp(0.5 * log_var)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Full VAE forward pass: encode → reparameterize → decode.

        Args:
            x:        Input image tensor of shape (B, 64, 64, 1).
            training: Whether in training mode.

        Returns:
            Tuple of:
                reconstructed: Reconstructed image (B, 64, 64, 1).
                mu:            Latent mean (B, latent_dim).
                log_var:       Latent log variance (B, latent_dim).
        """
        mu, log_var = self.encoder(x, training=training)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decoder(z, training=training)
        return reconstructed, mu, log_var

    def encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Encode images to (mu, log_var) — inference only.

        Args:
            x: Input image tensor of shape (B, 64, 64, 1).

        Returns:
            Tuple of (mu, log_var), each of shape (B, latent_dim).
        """
        return self.encoder(x, training=False)

    def decode(self, z: tf.Tensor) -> tf.Tensor:
        """
        Decode latent vectors to images — inference only.

        Args:
            z: Latent vector of shape (B, latent_dim).

        Returns:
            Reconstructed image tensor of shape (B, 64, 64, 1).
        """
        return self.decoder(z, training=False)

    def sample(self, n_samples: int = 16) -> tf.Tensor:
        """
        Generate new images by sampling from the prior N(0, I).

        This only works meaningfully for VAE, not AE, because
        the VAE's latent space is regularized to be close to N(0, I).

        Args:
            n_samples: Number of images to generate.

        Returns:
            Generated image tensor of shape (n_samples, 64, 64, 1).
        """
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(z)

    def get_config(self) -> dict:
        """Return model config for serialization."""
        return {"latent_dim": self.latent_dim}

    @classmethod
    def from_config(cls, config: dict) -> "VAE":
        """Reconstruct model from config."""
        return cls(**config)