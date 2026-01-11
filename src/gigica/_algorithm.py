import logging
import numpy as np
from scipy import linalg, stats
import numpy.linalg as nplinalg


def e_g(y: np.ndarray) -> float:
    # G(y) = log(cosh(y))
    # Use logaddexp for stability: log((e^y + e^-y)/2) = logaddexp(y, -y) - log(2)
    return (np.logaddexp(y, -y) - np.log(2)).mean()


def calc_gamma(e_g_y: float, e_g_v: float = 0.3745672075) -> float:
    # Constant for Negentropy J(w)
    # E[G(v)] for v ~ N(0,1) and G(y) = log(cosh(y))
    # Approximation ~ 0.3745672

    # gamma = E[G(y)] - E[G(v)]

    return e_g_y - e_g_v


def calc_k(ci: float, j: float) -> float:
    return (2 / np.pi) * np.arctan(ci * j)


def objective(alpha: float, ci: float, j: float, ref: np.ndarray, y: np.ndarray):
    val_K = calc_k(ci=ci, j=j)
    val_F = np.mean(y * ref)

    return alpha * val_K + (1 - alpha) * val_F


def negentropy(x: np.ndarray) -> float:
    e1 = e_g(x)
    e2 = 0.3745672075
    neg_J = (e1 - e2) ** 2
    # Handle edge cases
    if neg_J < np.finfo(np.float32).eps:
        neg_J = np.finfo(np.float64).eps  # Avoid division by zero

    return neg_J


try:
    from gigica._gigica import gig_ica_fit_rust
except ImportError as e:
    logging.warning(f"Failed to import Rust extension: {e}")
    gig_ica_fit_rust = None
except Exception as e:
    logging.warning(f"Failed to import Rust extension (Unknown Error): {e}")
    gig_ica_fit_rust = None


def gig_ica_fit(
    X: np.ndarray,
    references: np.ndarray,
    alpha: float = 0.5,
    whiten: bool = True,
    max_iter: int = 200,
    max_iter_line: int = 10,
    tol: float = 1e-5,
    mu: float = 1,
    rho: float = 0.5,
    beta: float = 0.02,
) -> np.ndarray:
    """
    Fit GIG-ICA to extract independent components guided by references.

    Tries to use the optimized Rust implementation if available.
    """
    if gig_ica_fit_rust is not None:
        try:
            return gig_ica_fit_rust(X, references, alpha, whiten, max_iter, tol)
        except Exception as e:
            logging.warning(
                f"failed during rust implementation {e}. Trying with python..."
            )
            # Fallback to python if rust fails for runtime reasons (e.g. dimension mismatch handled differently)
            pass

    return _gig_ica_fit_python(
        X, references, alpha, whiten, max_iter, max_iter_line, tol, mu, rho, beta
    )


def _gig_ica_fit_python(
    X: np.ndarray,
    references: np.ndarray,
    alpha: float = 0.5,
    whiten: bool = True,
    max_iter: int = 200,
    max_iter_line: int = 10,
    tol: float = 1e-5,
    mu: float = 1,
    rho: float = 0.5,
    beta: float = 0.02,
) -> np.ndarray:
    """
    Fit GIG-ICA to extract independent components guided by references.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Observed data (Time x Voxels).
    references : array-like, shape (n_components, n_features)
        Reference spatial maps.
    alpha : float, default=0.5
        Weighting parameter between independence and reference matching.
    whiten : bool, default=True
        Whether to whiten the data.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-5
        Tolerance for convergence.
    mu : float, default=1
        Initial step size for optimization (adjusted by line search)
    rho : float, default=0.5
        Factor by which step size will decrease during line search
    beta : float, default=0.02
        Factor for determining sufficient decrease in objective during line search

    Returns
    -------
    components : array, shape (n_components, n_features)
        The estimated independent components.
    """
    n_features = X.shape[1]
    n_components = references.shape[0]

    # Center data
    X_mean = X.mean(axis=1, keepdims=True)
    X -= X_mean

    # standardize reference
    z_references: np.ndarray = stats.zscore(references, axis=1, ddof=1)  # type:ignore

    # Whitening
    if whiten:
        # Use SVD for whitening: X = U * S * V^T
        # Covariance C = X X^T / n_features
        # C = U S^2 U^T / n_features
        # Whitening matrix K = inv(sqrt(C)). Actually better to use SVD of X directly.
        # X ~ (n_samples, n_features) where n_samples << n_features (Time << Voxels)
        # We want to whiten the temporal dimension.
        # X_white = K * X

        # SVD of X: U, S, Vt = svd(X)
        # X_white = sqrt(n_features) * U^T
        # Actually, let's follow standard ICA whitening.
        # C = np.dot(X, X.T) / n_features
        # w, v = linalg.eigh(C)
        # d = 1.0 / np.sqrt(w)
        # K = np.dot(v * d, v.T)
        # X_white = np.dot(K, X)

        # Using SVD for stability
        U, S, _ = nplinalg.svd(X, full_matrices=False)
        # K = U * diag(1/S) * U.T ?
        # Traditionally X_white = diag(1/S) * U.T * X = Vt (if we ignore variance scaling)
        # Let's retain unit variance
        K = np.dot(U, np.diag(1.0 / S)) * np.sqrt(n_features)
        del U, S
        X_white = np.dot(K.T, X)  # (n_samples, n_features)
        del K
        # X_white rows are uncorrelated and unit variance
    else:
        X_white = X

    X_white_pinv: np.ndarray = nplinalg.pinv(X_white)  # type:ignore (n_features, n_samples)

    components = np.zeros((n_components, n_features))

    for i in range(n_components):
        ref = z_references[i]

        # Initialize w
        # w_init = (ref * X_white_pinv.T).T ? No.
        # Eq: w0 = (R * X_white^+)^T
        # R (1, L), X_white^+ (L, M) -> (1, M) -> Transpose -> (M, 1)
        w = np.dot(ref, X_white_pinv).T
        w /= linalg.norm(w)

        # Yi = w^T * X_white
        y_est = np.dot(w.T, X_white)
        # J(w) = (E[G(y)] - E[G(v)])^2

        neg_J = negentropy(y_est)

        # F(w)
        val_F = np.mean(y_est * ref)
        # Ensure F is in [0, 1] range effectively by checking sign?
        # The paper says: "a special initialization of wi is adopted to make F(wi) range from 0 to 1"
        # Since we initialized w towards ref, correlation should be positive.

        # Calculate ci
        # K(w) = 2/pi * arctan(ci * J)
        # We want K(w0) = F(w0) (approximately, to balance scales)
        # tan(pi/2 * F) = ci * J
        # ci = tan(pi/2 * F) / J

        # Initial F might be close to 1
        initial_F = np.clip(val_F, 0, 0.999)  # Avoid tan(pi/2)

        if neg_J > np.finfo(np.float32).eps:
            ci = np.tan(np.pi / 2 * initial_F) / neg_J
        else:
            ci = 1.0

        # Optimization Loop
        for _ in range(max_iter):
            w_old = w.copy()

            # Forward pass to get y
            y_est = np.dot(w.T, X_white)  # (1, n_features)

            # Gradients
            # G(y) = log(cosh(y))
            E_Gy = e_g(y_est)

            gamma = calc_gamma(E_Gy)

            # J(w) = gamma^2
            val_J = gamma**2

            # g(y) = tanh(y)
            gy = np.tanh(y_est)

            # grad_J = 2 * gamma * E[X_white * g(y)]
            # X_white (M, L), g(y) (1, L)
            # E[...] -> mean over samples (L)
            # E[X * g] = (X * g^T) / L
            grad_J = 2 * gamma * np.mean(X_white * gy, axis=1)

            # grad_K
            # K = 2/pi * arctan(ci * J)
            # dK/dJ = 2/pi * ci / (1 + (ci*J)^2)
            dK_dJ = (2 / np.pi) * ci / (1 + (ci * val_J) ** 2)
            grad_K = dK_dJ * grad_J

            # grad_F
            # F = E[y * ref] = w^T * E[X * ref]
            # grad_F = E[X * ref]
            # grad_F = E[X * ref]
            grad_F = np.mean(X_white * ref, axis=1)

            # grad_C
            grad_C = alpha * grad_K + (1 - alpha) * grad_F

            # Direction
            grad_norm: float = linalg.norm(grad_C)
            if grad_norm < np.finfo(np.float32).eps:
                break
            direction = grad_C / grad_norm

            # Current Objective C(w)
            C_current = objective(alpha=alpha, ci=ci, j=val_J, y=y_est, ref=ref)

            # Simple line search (approximate Armijo)
            # Just check if objective increases? The paper suggests Armijo.
            # C(w_new) > C(w) + beta * mu * grad_C.T * direction
            # direction is normalized gradient, so grad.T * dir = norm(grad)

            # Line search (backtracking)
            mu = 1.0  # Initial step size
            improved = False
            w_new = w

            # note that this is slightly different than the paper describes,
            # in that we're estimating the step size (mu) for every optimization iteration.
            # The paper just suggests updating mu at most once per iteration
            for _ in range(max_iter_line):
                w_try = w + mu * direction
                w_try /= linalg.norm(w_try)

                y_try = np.dot(w_try.T, X_white)

                # Compute C_new
                gamma_try = calc_gamma(e_g(y_try))
                C_try = objective(alpha=alpha, ci=ci, j=gamma_try**2, ref=ref, y=y_try)

                # Check condition
                # slope = grad_C.T * direction = norm(grad_C)
                threshold = C_current + beta * mu * grad_norm

                if C_try > threshold:
                    w_new = w_try
                    # mu_final = mu
                    improved = True
                    break
                else:
                    mu *= rho

            if improved:
                w = w_new
            else:
                # If cannot improve, maybe step size too small or converged
                break

            # Check convergence
            if 1 - np.abs(np.dot(w.T, w_old)).item() < tol:
                break
        else:
            logging.warning(
                f"GIG-ICA did not converge for component {i} after {max_iter} iterations."
            )

        # Store component
        components[i, :] = np.dot(w.T, X_white)

    return components
