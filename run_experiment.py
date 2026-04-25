"""
run_experiment.py
-----------------
Master script that runs the full experiment pipeline:
    1. Train all 12 models (6 AE + 6 VAE, one per anatomical region)
    2. Generate all visualizations and save to figures/

Usage:
    python run_experiment.py
"""

from src.train import train_all
from src.utils import save_all_visualizations

if __name__ == "__main__":
    # Step 1: Train all models
    results = train_all(raw_data_dir="data/raw", epochs=15)

    # Step 2: Generate all visualizations
    save_all_visualizations(results)

    print("\nExperiment complete.")
    print("  Models  → models/")
    print("  Figures → figures/")