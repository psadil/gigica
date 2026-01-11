import numpy as np
import pytest
from gigica import GIGICA

from tests.utils import generate_synthetic_data


def test_low_samples_high_features():
    """Test GIG-ICA when n_samples (time) << n_features (voxels)."""

    n_samples = 50
    n_features = 1000
    n_components = 3

    X, references, S_true, _ = generate_synthetic_data(n_samples, n_features, n_components)

    gig = GIGICA(alpha=0.5, max_iter=100, tol=1e-4)
    gig.fit(X, references=references)

    # Verify shape of extracted components
    assert gig.components_.shape == (n_components, n_features)

    # Verify correlations with ground truth
    extracted = gig.components_
    for i in range(n_components):
        # Calculate correlation with corresponding ground truth
        # Note: GIG-ICA matches references order, so i-th component should match i-th source
        corr = np.corrcoef(extracted[i], S_true[i])[0, 1]
        print(f"Component {i} correlation: {corr}")
        assert np.abs(corr) > 0.9, f"Component {i} poorly recovered"


def test_rank_deficient_input():
    """Test GIG-ICA when input is rank deficient."""
    n_samples = 50
    n_features = 1000
    n_components = 3

    # Rank 1 mixing matrix
    A = np.ones((n_samples, n_components))
    S = np.random.randn(n_components, n_features)
    X = np.dot(A, S)  # Rank 1 effectively (if S uncorrelated, X cols are scaled copies)

    # Actually rank of X is min(rank(A), rank(S)) = 1.

    references = S + 0.1 * np.random.randn(n_components, n_features)

    # This might struggle or produce degenerate outputs, but shouldn't crash
    gig = GIGICA(alpha=0.5)
    try:
        gig.fit(X, references=references)
    except Exception as e:
        pytest.fail(f"Fit failed with rank deficient input: {e}")


if __name__ == "__main__":
    test_low_samples_high_features()
    test_rank_deficient_input()
    print("All tests passed!")
