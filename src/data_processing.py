"""
data_processing.py
------------------
Handles all dataset loading, preprocessing, and splitting for the
Medical MNIST dataset using tf.data pipelines.

Each anatomical region is processed independently, producing
separate train/val/test datasets per region.
"""

import os
import pathlib
from typing import Dict, List, Tuple

import tensorflow as tf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_SIZE: Tuple[int, int] = (64, 64)
IMAGE_CHANNELS: int = 1          # Medical MNIST images are grayscale
BATCH_SIZE: int = 32
AUTOTUNE = tf.data.AUTOTUNE

ANATOMICAL_REGIONS: List[str] = [
    "AbdomenCT",
    "BreastMRI",
    "ChestCT",
    "CXR",
    "Hand",
    "HeadCT",
]

# Train / Validation / Test split ratios (must sum to 1.0)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_and_preprocess(image_path: tf.Tensor) -> tf.Tensor:
    """
    Load a single image from disk and preprocess it.

    Steps:
        1. Read raw bytes from disk.
        2. Decode JPEG into a uint8 tensor.
        3. Resize to IMAGE_SIZE.
        4. Convert to float32 and normalize to [0, 1].
        5. Ensure shape is (H, W, 1) — grayscale channel last.

    Args:
        image_path: Scalar string tensor containing the file path.

    Returns:
        Preprocessed image tensor of shape (64, 64, 1), dtype float32.
    """
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=IMAGE_CHANNELS)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _build_dataset_from_paths(
    paths: List[str],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Build a batched tf.data.Dataset from a list of image file paths.

    Args:
        paths:      List of absolute file path strings.
        batch_size: Number of images per batch.
        shuffle:    Whether to shuffle the dataset.

    Returns:
        A tf.data.Dataset yielding batches of shape (B, 64, 64, 1).
    """
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle:
        path_ds = path_ds.shuffle(
            buffer_size=len(paths),
            reshuffle_each_iteration=True,
            seed=42,
        )

    image_ds = path_ds.map(
        _load_and_preprocess,
        num_parallel_calls=AUTOTUNE,
    )

    return (
        image_ds
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )


def _split_paths(
    paths: List[str],
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split a list of file paths into train, validation, and test subsets.

    Uses a fixed seed for reproducibility. Ratios are defined by
    TRAIN_RATIO, VAL_RATIO, and TEST_RATIO module constants.

    Args:
        paths: Full list of file paths for one anatomical region.

    Returns:
        Tuple of (train_paths, val_paths, test_paths).
    """
    # Shuffle with fixed seed before splitting — ensures reproducibility
    import random
    rng = random.Random(42)
    paths = list(paths)
    rng.shuffle(paths)

    n = len(paths)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    train_paths = paths[:train_end]
    val_paths = paths[train_end:val_end]
    test_paths = paths[val_end:]

    return train_paths, val_paths, test_paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_image_paths_per_region(
    raw_data_dir: str,
) -> Dict[str, List[str]]:
    """
    Scan the raw data directory and collect image paths per region.

    Expects the following structure:
        raw_data_dir/
            AbdomenCT/   ← folder name must match ANATOMICAL_REGIONS
            BreastMRI/
            ...

    Args:
        raw_data_dir: Path to the raw dataset directory.

    Returns:
        Dictionary mapping region name → list of image file paths.

    Raises:
        FileNotFoundError: If raw_data_dir does not exist.
        ValueError: If no images are found for a region.
    """
    data_dir = pathlib.Path(raw_data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_data_dir}"
        )

    region_paths: Dict[str, List[str]] = {}

    for region in ANATOMICAL_REGIONS:
        region_dir = data_dir / region

        if not region_dir.exists():
            raise FileNotFoundError(
                f"Expected folder for region '{region}' at: {region_dir}"
            )

        paths = [
            str(p) for p in region_dir.glob("*.jpeg")
        ] + [
            str(p) for p in region_dir.glob("*.jpg")
        ] + [
            str(p) for p in region_dir.glob("*.png")
        ]

        if not paths:
            raise ValueError(
                f"No images found in {region_dir}. "
                "Check your dataset extraction."
            )

        region_paths[region] = paths
        print(f"  [{region}] → {len(paths)} images found")

    return region_paths


def build_region_datasets(
    raw_data_dir: str,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Dict[str, tf.data.Dataset]]:
    """
    Build train/val/test tf.data.Dataset objects for every region.

    This is the main entry point for data loading. Call this function
    from your training script to get all datasets ready for use.

    Args:
        raw_data_dir: Path to the raw dataset directory.
        batch_size:   Number of images per batch (default: 32).

    Returns:
        Nested dictionary with structure:
            {
                "AbdomenCT": {
                    "train": tf.data.Dataset,
                    "val":   tf.data.Dataset,
                    "test":  tf.data.Dataset,
                    "n_train": int,
                    "n_val":   int,
                    "n_test":  int,
                },
                "BreastMRI": { ... },
                ...
            }

    Example:
        >>> datasets = build_region_datasets("data/raw")
        >>> for image_batch in datasets["AbdomenCT"]["train"]:
        ...     print(image_batch.shape)  # (32, 64, 64, 1)
    """
    print("\n=== Loading Medical MNIST Dataset ===")
    region_paths = get_image_paths_per_region(raw_data_dir)

    all_datasets: Dict[str, Dict] = {}

    for region, paths in region_paths.items():
        train_paths, val_paths, test_paths = _split_paths(paths)

        train_ds = _build_dataset_from_paths(
            train_paths, batch_size, shuffle=True
        )
        val_ds = _build_dataset_from_paths(
            val_paths, batch_size, shuffle=False
        )
        test_ds = _build_dataset_from_paths(
            test_paths, batch_size, shuffle=False
        )

        all_datasets[region] = {
            "train":   train_ds,
            "val":     val_ds,
            "test":    test_ds,
            "n_train": len(train_paths),
            "n_val":   len(val_paths),
            "n_test":  len(test_paths),
        }

        print(
            f"  [{region}] split → "
            f"train: {len(train_paths)} | "
            f"val: {len(val_paths)} | "
            f"test: {len(test_paths)}"
        )

    print("=== Dataset Ready ===\n")
    return all_datasets