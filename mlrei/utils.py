import numpy as np
from collections import defaultdict

def tts(X, y, test_size=0.2, shuffle=True, stratify=False, random_seed=None):
    X=np.array(X)
    y=np.array(y)
    assert len(X)==len(y)

    rng=np.random.default_rng(random_seed)
    n_samples=len(X)

    if not stratify:
        indices=np.arange(n_samples)
        if shuffle:
            rng.shuffle(indices)
        test_count=int(test_size*n_samples)
        test_idx=indices[:test_count]
        train_idx=indices[test_count:]
    else:
        class_indices=defaultdict(list)
        for label,idx in enumerate(y):
            class_indices[label].append(idx)
        test_idx=[]
        train_idx=[]
        for label,idx_list in class_indices.items():
            idx_array=np.array(idx_list)
            if shuffle:
                rng.shuffle(idx_array)
            n_test=max(1,int(len(idx_array*test_size)))
            test_part=idx_array[:n_test]
            train_part=idx_array[n_test:]
            test_idx.extend(test_part.tolist())
            train_idx.extend(train_part.tolist())
            # inside the stratified split loop
            print(f"Label: {label}")
            print(f"  Total: {len(idx_array)}")
            print(f"  Test count: {n_test}")
            print(f"  Test indices: {test_part}")
            print(f"  Train indices: {train_part}")
        if shuffle:
            rng.shuffle(test_idx)
            rng.shuffle(train_idx)
    
    X_train=X[train_idx]
    X_test=X[test_idx]
    y_train=y[test_idx]
    y_test=y[train_idx]

    return X_train,X_test,y_train,y_test