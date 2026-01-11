import numpy as np
from gigica import GIGICA
from tests.utils import generate_synthetic_data


def test_transform_recovery():
    """Test GIGICA.transform() recovers mixing matrix (time courses)."""
    n_samples = 200
    n_features = 1000
    n_components = 5

    # X (200, 1000) = A (200, 5) * S (5, 1000)
    X, references, S_true, A_true = generate_synthetic_data(
        n_samples, n_features, n_components
    )

    gig = GIGICA(alpha=0.5, tol=1e-5)
    gig.fit(X, references=references)

    # Transform to recover mixing matrix A
    A_est = gig.transform(X)

    assert A_est.shape == (n_samples, n_components)

    # Verify A_est matches A_true
    # Since components are guided, order should be preserved.
    print("\nMixing Matrix Correlation:")
    for i in range(n_components):
        # Calculate correlation between estimated and true time course
        corr = np.corrcoef(A_est[:, i], A_true[:, i])[0, 1]
        print(f"Component {i} mixing correlation: {corr}")

        # We expect high correlation
        # Sign might be flipped, but generally GIGICA tries to match reference sign.
        # References are +ve correlated with S_true. So S_est should be +ve correlated.
        # So A_est should be +ve correlated with A_true.
        assert corr > 0.9, f"Mixing component {i} poorly recovered (corr={corr})"


if __name__ == "__main__":
    test_transform_recovery()
