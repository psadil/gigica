import pytest
import logging
from gigica._algorithm import _gig_ica_fit_python
from tests.utils import generate_synthetic_data


def test_convergence_warning_python(caplog):
    """Test that Python implementation logs warning when not converging."""
    n_samples = 100
    n_features = 500
    n_components = 2
    X, references, _, _ = generate_synthetic_data(n_samples, n_features, n_components)

    # Force non-convergence with max_iter=0
    max_iter = 1

    with caplog.at_level(logging.WARNING):
        _gig_ica_fit_python(X, references, max_iter=max_iter)

    assert "did not converge" in caplog.text
    assert f"after {max_iter} iterations" in caplog.text


def test_convergence_warning_rust(caplog):
    """Test that Rust implementation logs warning when not converging."""
    try:
        from gigica._gigica import gig_ica_fit_rust
    except ImportError:
        pytest.skip("Rust implementation not available")

    n_samples = 100
    n_features = 500
    n_components = 2
    X, references, _, _ = generate_synthetic_data(n_samples, n_features, n_components)

    # Force non-convergence with max_iter=0
    max_iter = 1

    with caplog.at_level(logging.WARNING):
        # Note: calling the rust function directly via the GIGICA estimator or directly
        # gig_ica_fit_rust(py, x, references, alpha, whiten, max_iter, tol)
        # We call it as python function: (X, references, alpha, whiten, max_iter, tol)
        gig_ica_fit_rust(X, references, 0.5, True, max_iter, 1e-6)

    assert "did not converge" in caplog.text
    assert f"after {max_iter} iterations" in caplog.text


if __name__ == "__main__":
    pytest.main([__file__])
