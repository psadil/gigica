import numpy as np
from sklearn import decomposition
from gigica.estimator import GIGICA

from tests.utils import generate_synthetic_data


def test_recovery():
    # Generate synthetic data for Temporal ICA
    # We want independence over "Time" (Features=2000).
    # Channels=3 (Samples=3).
    n_channels = 3
    n_timepoints = 2000
    n_components = 3

    # X: (Channels, Time). S_true: (Components, Time).
    X, references, S_true, _ = generate_synthetic_data(
        n_channels, n_timepoints, n_components
    )

    # Run FastICA
    # FastICA expects (Samples, Features) where independence is over columns (?)
    # Actually Scikit FastICA expects (n_samples, n_features). It decomposes X = S * A.
    # Returns S (n_samples, n_components). Independence over "distribution of values in columns".
    # Here our "Samples" are Timepoints. Our "Features" are Channels.
    # So we want to feed (2000, 3).
    X_fast = X.T

    fast_ica = decomposition.FastICA(n_components=n_components, random_state=42)
    S_fast = fast_ica.fit_transform(X_fast)  # (2000, 3)

    # Run GIG-ICA
    # GIG-ICA expects (Samples, Features). Independence over Features (Columns).
    # Here inputs X is (Channels, Time). Independent components are Time courses (Rows of S).
    # GIGICA recovers S (Components, Features) = (3, 2000).
    # This matches our X.

    gig_ica = GIGICA(alpha=0.5)
    gig_ica.fit(X, references=references)

    # components_ is (Components, Features) = (3, 2000)
    # This matches S_true
    S_gig = gig_ica.components_

    # Evaluate correlation with Ground Truth S
    print("\nCorrelations with Ground Truth:")

    # S_true is (3, 2000)
    # S_fast is (2000, 3) -> Transpose to (3, 2000)
    S_fast_T = S_fast.T

    corr_fast = []
    corr_gig = []

    for i in range(3):
        # Ground truth component i (Row)
        gt = S_true[i, :]

        # Best match in FastICA
        corrs = [np.abs(np.corrcoef(gt, S_fast_T[j, :])[0, 1]) for j in range(3)]
        corr_fast.append(max(corrs))

        # Best match in GIG-ICA
        corrs_g = [np.abs(np.corrcoef(gt, S_gig[j, :])[0, 1]) for j in range(3)]
        corr_gig.append(max(corrs_g))

    print(f"FastICA Correlations: {corr_fast}")
    print(f"GIG-ICA Correlations: {corr_gig}")

    assert (
        np.mean(corr_gig) > np.mean(corr_fast) - 0.1
    ), "GIG-ICA performed significantly worse!"
