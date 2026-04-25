"""
test_data_processing.py
-----------------------
Unit tests for the data_processing module.
"""

import tensorflow as tf
import pytest
from src.data_processing import (
    build_region_datasets,
    ANATOMICAL_REGIONS,
    IMAGE_SIZE,
)


class TestBuildRegionDatasets:
    """Tests for build_region_datasets()."""

    @pytest.fixture(scope="class")
    def datasets(self):
        """Load datasets once for all tests in this class."""
        return build_region_datasets("data/raw")

    def test_all_regions_present(self, datasets):
        """All 6 anatomical regions must be in the output."""
        for region in ANATOMICAL_REGIONS:
            assert region in datasets, f"Missing region: {region}"

    def test_all_splits_present(self, datasets):
        """Each region must have train, val, and test splits."""
        for region in ANATOMICAL_REGIONS:
            for split in ("train", "val", "test"):
                assert split in datasets[region], (
                    f"Missing split '{split}' for {region}"
                )

    def test_batch_shape(self, datasets):
        """Each batch must have shape (B, 64, 64, 1)."""
        for batch in datasets["AbdomenCT"]["train"].take(1):
            assert batch.shape[1:] == (64, 64, 1), (
                f"Unexpected shape: {batch.shape}"
            )

    def test_pixel_range(self, datasets):
        """Pixel values must be normalized to [0, 1]."""
        for batch in datasets["AbdomenCT"]["test"].take(1):
            assert float(tf.reduce_min(batch)) >= 0.0
            assert float(tf.reduce_max(batch)) <= 1.0

    def test_split_counts_positive(self, datasets):
        """All split counts must be greater than zero."""
        for region in ANATOMICAL_REGIONS:
            assert datasets[region]["n_train"] > 0
            assert datasets[region]["n_val"] > 0
            assert datasets[region]["n_test"] > 0

    def test_train_larger_than_val(self, datasets):
        """Training set must be larger than validation set."""
        for region in ANATOMICAL_REGIONS:
            assert datasets[region]["n_train"] > datasets[region]["n_val"]