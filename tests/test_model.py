"""
test_model.py
-------------
Unit tests for the AE and VAE model architectures.
"""

import tensorflow as tf
import pytest
from src.model import Autoencoder, VAE, LATENT_DIM


BATCH_SIZE = 4
DUMMY_INPUT = tf.random.normal((BATCH_SIZE, 64, 64, 1))


class TestAutoencoder:
    """Tests for the Autoencoder model."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create one AE instance for all tests."""
        return Autoencoder(latent_dim=LATENT_DIM)

    def test_output_shape(self, model):
        """AE output must match input shape."""
        output = model(DUMMY_INPUT)
        assert output.shape == DUMMY_INPUT.shape, (
            f"Expected {DUMMY_INPUT.shape}, got {output.shape}"
        )

    def test_output_range(self, model):
        """AE output must be in [0, 1] due to sigmoid activation."""
        output = model(DUMMY_INPUT)
        assert float(tf.reduce_min(output)) >= 0.0
        assert float(tf.reduce_max(output)) <= 1.0

    def test_encode_shape(self, model):
        """Encoder must output (B, latent_dim)."""
        z = model.encode(DUMMY_INPUT)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_decode_shape(self, model):
        """Decoder must output (B, 64, 64, 1)."""
        z = tf.random.normal((BATCH_SIZE, LATENT_DIM))
        output = model.decode(z)
        assert output.shape == (BATCH_SIZE, 64, 64, 1)

    def test_trainable_variables_exist(self, model):
        """Model must have trainable parameters."""
        _ = model(DUMMY_INPUT)  # build the model
        assert len(model.trainable_variables) > 0


class TestVAE:
    """Tests for the Variational Autoencoder model."""

    @pytest.fixture(scope="class")
    def model(self):
        """Create one VAE instance for all tests."""
        return VAE(latent_dim=LATENT_DIM)

    def test_output_shapes(self, model):
        """VAE must return (reconstructed, mu, log_var) with correct shapes."""
        reconstructed, mu, log_var = model(DUMMY_INPUT)
        assert reconstructed.shape == DUMMY_INPUT.shape
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert log_var.shape == (BATCH_SIZE, LATENT_DIM)

    def test_output_range(self, model):
        """Reconstructed output must be in [0, 1]."""
        reconstructed, _, _ = model(DUMMY_INPUT)
        assert float(tf.reduce_min(reconstructed)) >= 0.0
        assert float(tf.reduce_max(reconstructed)) <= 1.0

    def test_reparameterize_shape(self, model):
        """Reparameterization must return correct shape."""
        mu = tf.zeros((BATCH_SIZE, LATENT_DIM))
        log_var = tf.zeros((BATCH_SIZE, LATENT_DIM))
        z = model.reparameterize(mu, log_var)
        assert z.shape == (BATCH_SIZE, LATENT_DIM)

    def test_reparameterize_stochastic(self, model):
        """Two calls to reparameterize must return different samples."""
        mu = tf.zeros((BATCH_SIZE, LATENT_DIM))
        log_var = tf.zeros((BATCH_SIZE, LATENT_DIM))
        z1 = model.reparameterize(mu, log_var)
        z2 = model.reparameterize(mu, log_var)
        assert not tf.reduce_all(tf.equal(z1, z2)), (
            "Reparameterization should be stochastic"
        )

    def test_sample_shape(self, model):
        """VAE sample() must return correct shape."""
        _ = model(DUMMY_INPUT)  # build first
        samples = model.sample(n_samples=8)
        assert samples.shape == (8, 64, 64, 1)

    def test_encode_returns_mu_and_logvar(self, model):
        """encode() must return (mu, log_var) tuple."""
        mu, log_var = model.encode(DUMMY_INPUT)
        assert mu.shape == (BATCH_SIZE, LATENT_DIM)
        assert log_var.shape == (BATCH_SIZE, LATENT_DIM)

    def test_trainable_variables_exist(self, model):
        """VAE must have trainable parameters."""
        _ = model(DUMMY_INPUT)
        assert len(model.trainable_variables) > 0