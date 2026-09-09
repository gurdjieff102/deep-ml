import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    n_samples = X.shape[0]

    if replacements:
        subset_size = n_samples
    else:
        subset_size = n_samples // 2

    subsets = []

    for _ in range(n_subsets):
        indices = np.random.choice(
            n_samples,
            size=subset_size,
            replace=replacements
        )

        X_subset = X[indices]
        y_subset = y[indices]

        subsets.append((
            X_subset.tolist(),
            y_subset.tolist()
        ))

    return subsets