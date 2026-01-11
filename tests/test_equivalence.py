import numpy as np
import pytest
from gigica._algorithm import _gig_ica_fit_python

from tests.utils import generate_synthetic_data

try:
    from gigica._gigica import gig_ica_fit_rust
except ImportError:
    gig_ica_fit_rust = None


def test_rust_python_equivalence():
    if gig_ica_fit_rust is None:
        pytest.skip("Rust implementation not available")

    # Generate synthetic data
    n_samples = 200
    n_features = 1000
    n_components = 5

    X, references, _, _ = generate_synthetic_data(n_samples, n_features, n_components)

    # Common parameters
    alpha = 0.5
    whiten = True
    max_iter = 100
    tol = 1e-6

    # Run Python implementation
    # Note: _gig_ica_fit_python signature:
    # (X, references, alpha=0.5, whiten=True, max_iter=200, max_iter_line=10, tol=1e-5, mu=1, rho=0.5, beta=0.02)
    components_python = _gig_ica_fit_python(
        X, references, alpha=alpha, whiten=whiten, max_iter=max_iter, tol=tol
    )

    # Run Rust implementation
    # gig_ica_fit_rust signature from lib.rs:
    # (py, x, references, alpha, whiten, max_iter, tol)
    components_rust = gig_ica_fit_rust(
        X, references, alpha=alpha, whiten=whiten, max_iter=max_iter, tol=tol
    )

    # Verify shapes
    assert components_python.shape == components_rust.shape

    # Compare results
    # Since GIG-ICA is reference-guided, the ambiguity of sign and permutation
    # should be largely resolved, and they should match closely.

    # Check correlation between corresponding components
    for i in range(n_components):
        # Pearson correlation
        corr = np.corrcoef(components_python[i], components_rust[i])[0, 1]
        print(f"Component {i} correlation: {corr}")
        assert (
            corr > 0.99
        ), f"Component {i} does not match between Python and Rust (corr={corr})"


if __name__ == "__main__":
    test_rust_python_equivalence()
