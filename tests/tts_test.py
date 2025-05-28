# tests/tts.py

import numpy as np
import pytest

from mlrei.utils import tts


def test_split_sizes_without_shuffle():
    # Simple sequential split (shuffle=False)
    X = np.arange(10).reshape(10, 1)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, shuffle=False)

    # 10 samples → test_size=0.3 → 3 test, 7 train
    # test_split_sizes_without_shuffle in tests/tts.py

    assert X_train.shape[0] == 7
    assert X_test.shape[0] == 3
    np.testing.assert_array_equal(y_train, y[3:])
    np.testing.assert_array_equal(y_test, y[:3])



def test_reproducibility_with_seed():
    # Ensure same seed gives identical splits
    X = np.arange(20).reshape(20, 1)
    y = np.arange(20) % 2  # alternating 0/1 labels

    out1 = tts(X, y, test_size=0.5, shuffle=True, random_seed=42)
    out2 = tts(X, y, test_size=0.5, shuffle=True, random_seed=42)

    for a, b in zip(out1, out2):
        np.testing.assert_array_equal(a, b)


def test_stratified_split_preserves_class_ratios():
    # Imbalanced dataset: 7 zeros, 3 ones
    X = np.array([[i] for i in range(10)])
    y = np.array([0] * 7 + [1] * 3)

    # 30% test → expect:
    #   zeros: max(1, int(7 * 0.3)) = max(1, 2) = 2
    #   ones:  max(1, int(3 * 0.3)) = max(1, 0) = 1
    X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.3, shuffle=True, stratify=True, random_seed=0
    )

    # Check test-set class counts
    zeros_test = list(y_test).count(0)
    ones_test = list(y_test).count(1)
    assert zeros_test == 2
    assert ones_test == 1

    # Check train-set class counts (remaining samples)
    zeros_train = list(y_train).count(0)
    ones_train = list(y_train).count(1)
    assert zeros_train == 5
    assert ones_train == 2