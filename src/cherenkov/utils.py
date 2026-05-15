import numpy as np

def normalize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization per feature."""
    return (X - X.mean(0)) / (X.std(0) + 1e-8)
