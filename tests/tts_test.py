# tests/tts.py

"""
Pytest test suite for validating the tts() function in mlrei.utils.
"""

import numpy as np
import pytest
from mlrei.utils import tts

def test_basic_split():
    """
    Test basic train-test split without stratification and without shuffling.
    Ensures correct partitioning and order preservation.
    """
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)

    # Perform deterministic split (no shuffle)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, shuffle=False)

    # Check split sizes
    assert X_train.shape[0] == 7
    assert X_test.shape[0] == 3

    # Without shuffle: test should be first 3, train should be last 7
    np.testing.assert_array_equal(y_test, np.array([0, 1, 2]))
    np.testing.assert_array_equal(y_train, np.array([3, 4, 5, 6, 7, 8, 9]))


def test_stratified_split():
    """
    Test that stratified splitting preserves class distributions approximately.
    Uses an imbalanced dataset (7 zeros, 3 ones).
    """
    X = np.array([[i] for i in range(10)])
    y = np.array([0] * 7 + [1] * 3)

    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.3, shuffle=True, stratify=True, random_seed=42
    )

    # Expected test class distribution:
    # Class 0: max(1, int(7 * 0.3)) = 2
    # Class 1: max(1, int(3 * 0.3)) = 1
    assert (y_test == 0).sum() == 2
    assert (y_test == 1).sum() == 1

    # Expected train class distribution: remaining samples
    assert (y_train == 0).sum() == 5
    assert (y_train == 1).sum() == 2


def test_reproducibility():
    """
    Test that using the same random seed produces identical splits.
    """
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20) % 2  # alternating labels

    # Two splits with the same seed
    result1 = tts(X, y, test_size=0.5, shuffle=True, random_seed=42)
    result2 = tts(X, y, test_size=0.5, shuffle=True, random_seed=42)

    # All returned arrays should be identical
    for a, b in zip(result1, result2):
        np.testing.assert_array_equal(a, b)


def test_edge_cases():
    """
    Test edge conditions:
    - Very small datasets
    - Multiple test_size values
    """
    # Very small dataset
    X_small = np.array([[1], [2]])
    y_small = np.array([0, 1])

    X_train, X_test, y_train, y_test = tts(X_small, y_small, test_size=0.5, shuffle=False)
    assert len(X_train) == 1
    assert len(X_test) == 1

    # Test a variety of valid test_size values
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100) % 3  # 3 classes

    for test_size in [0.1, 0.2, 0.3, 0.5]:
        X_tr, X_te, y_tr, y_te = tts(X, y, test_size=test_size, shuffle=False)
        expected_test = int(100 * test_size)
        expected_train = 100 - expected_test

        assert len(X_te) == expected_test
        assert len(X_tr) == expected_train