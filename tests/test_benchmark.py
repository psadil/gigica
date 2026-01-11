import time
from tests.utils import generate_synthetic_data
from gigica._algorithm import gig_ica_fit_rust, _gig_ica_fit_python


def test_speedup():
    # Generate synthetic data
    n_samples = 1000
    n_features = 10000
    n_components = 50

    X, references, _, _ = generate_synthetic_data(n_samples, n_features, n_components)

    print(
        f"Data: {n_samples} timepoints x {n_features} voxels, {n_components} components."
    )

    # Bench Python
    start_py = time.time()
    _gig_ica_fit_python(
        X, references, max_iter=50
    )  # Reduced iterations for benchmark speed
    end_py = time.time()
    time_py = end_py - start_py
    print(f"Python time: {time_py:.4f}s")

    # Bench Rust
    if gig_ica_fit_rust is None:
        print("Rust implementation not available!")
        return

    start_rust = time.time()
    gig_ica_fit_rust(X, references, max_iter=50, alpha=0.5, whiten=True, tol=1e-5)
    end_rust = time.time()
    time_rust = end_rust - start_rust
    print(f"Rust time: {time_rust:.4f}s")

    assert time_rust < (time_py * 0.75)
