import numpy as np
from collections import defaultdict

def tts(X, y, test_size=0.2, shuffle=True, stratify=False, random_seed=None):
    """
    Custom implementation of a train-test split function with optional stratification and shuffling.

    Parameters
    ----------
    X : array-like
        Feature dataset, where each row is a sample and each column is a feature.
    y : array-like
        Target labels corresponding to each sample in X.
    test_size : float, optional (default=0.2)
        Proportion of the dataset to include in the test split. Must be between 0 and 1 (exclusive).
    shuffle : bool, optional (default=True)
        Whether to shuffle the data before splitting. Useful to ensure randomness.
    stratify : bool, optional (default=False)
        Whether to preserve the class distribution in train and test splits. If True, `y` must be categorical.
    random_seed : int or None, optional (default=None)
        Seed value for reproducibility. Passed to NumPy’s random number generator.

    Returns
    -------
    X_train : ndarray
        Training split of features.
    X_test : ndarray
        Test split of features.
    y_train : ndarray
        Training split of labels.
    y_test : ndarray
        Test split of labels.

    Raises
    ------
    AssertionError
        If `X` and `y` lengths don't match, or if `test_size` is not in the range (0, 1).

    Notes
    -----
    - If `stratify=True`, the function ensures that the class proportions in `y` are approximately preserved in the
      train and test sets. This is helpful for imbalanced datasets.
    - The split is deterministic when `random_seed` is set.
    """

    # Convert inputs to NumPy arrays for consistent indexing
    X = np.array(X)
    y = np.array(y)

    # Basic validations
    assert len(X) == len(y), "X and y must have the same length"
    assert 0 < test_size < 1, "test_size must be between 0 and 1 (exclusive)"

    # Create a random number generator for reproducibility
    rng = np.random.default_rng(random_seed)
    n_samples = len(X)

    if not stratify:
        # If not stratifying, just split randomly or sequentially
        indices = np.arange(n_samples)

        # Shuffle indices if required
        if shuffle:
            rng.shuffle(indices)

        # Determine the number of test samples
        test_count = int(test_size * n_samples)

        # First part for test, rest for train
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]

    else:
        # For stratified split, we ensure the same class distribution in train and test
        class_indices = defaultdict(list)

        # Group indices by class label
        for idx, label in enumerate(y):
            class_indices[label].append(idx)

        test_idx = []
        train_idx = []

        # For each class, split its indices
        for label, idx_list in class_indices.items():
            idx_array = np.array(idx_list)

            # Shuffle class indices if required
            if shuffle:
                rng.shuffle(idx_array)

            # Ensure at least one test sample per class
            n_test = max(1, int(len(idx_array) * test_size))

            # Split class indices into test and train
            test_part = idx_array[:n_test]
            train_part = idx_array[n_test:]

            # Collect indices across all classes
            test_idx.extend(test_part.tolist())
            train_idx.extend(train_part.tolist())

        # Shuffle final test/train indices if needed
        if shuffle:
            rng.shuffle(test_idx)
            rng.shuffle(train_idx)

        # Convert to NumPy arrays for indexing
        test_idx = np.array(test_idx)
        train_idx = np.array(train_idx)

    # Use the computed indices to split the data
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test