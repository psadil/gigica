import numpy as np
from numpy import random


def generate_synthetic_data(
    n_samples=200, n_features=1000, n_components=5, noise_level=0.05, random_seed=42
):
    """
    Generates synthetic fMRI-like data for GIG-ICA testing.

    Returns:
        X: (n_samples, n_features) Observed mixed signal
        references: (n_components, n_features) Noisy reference signals
        S_true: (n_components, n_features) Ground truth sources
    """
    rs = random.default_rng(random_seed)

    # True sources (spatial maps): Super-Gaussian (Laplace) distributions are typical for ICA
    S_true = rs.laplace(size=(n_components, n_features))

    # Mixing matrix (time courses): Gaussian
    A_true = rs.normal(size=(n_samples, n_components))

    # Generated signal (Time x Space)
    # X = A * S
    # Add noise to ensure full rank and simulate real data
    X = np.dot(A_true, S_true) + noise_level * rs.normal(size=(n_samples, n_features))

    # References (noisy version of true sources)
    # Simulates group-level components used as priors
    references = S_true + 0.5 * rs.normal(size=S_true.shape)

    return X, references, S_true, A_true
