"""
fix_models.py
-------------
Re-saves all trained models using load_weights instead of load_model,
bypassing the config deserialization issue.
"""

import tensorflow as tf
from src.model import Autoencoder, VAE
from src.data_processing import build_region_datasets, ANATOMICAL_REGIONS

datasets = build_region_datasets("data/raw")

for region in ANATOMICAL_REGIONS:
    print(f"Re-saving {region}...")

    # Build fresh models by passing a real batch (required before load_weights)
    ae  = Autoencoder(latent_dim=2)
    vae = VAE(latent_dim=2)

    for batch in datasets[region]["train"].take(1):
        ae(batch)
        vae(batch)

    # Load weights directly — bypasses config deserialization entirely
    ae.load_weights(f"models/{region}_ae_v1.keras")
    vae.load_weights(f"models/{region}_vae_v1.keras")

    # Re-save now that get_config/from_config are defined in model.py
    ae.save(f"models/{region}_ae_v1.keras")
    vae.save(f"models/{region}_vae_v1.keras")

    print(f"  [{region}] done")

print("\nAll models re-saved.")