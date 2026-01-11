from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import validation
import numpy as np
from scipy import linalg
from gigica import _algorithm


class GIGICA(BaseEstimator, TransformerMixin):
    """
    Group-Information Guided Independent Component Analysis (GIG-ICA).

    Parameters
    ----------
    alpha : float, default=0.5
        Weighting parameter between independence (0) and reference matching (1).
    whiten : bool, default=True
        Whether to whiten the data.
    max_iter : int, default=200
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        The estimated independent components (spatial maps).
    references_ : array, shape (n_components, n_features)
        The reference components used.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        whiten: bool = True,
        max_iter: int = 200,
        tol: float = 1e-4,
    ):
        self.alpha = alpha
        self.whiten = whiten
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: None = None, references: np.ndarray | None = None):
        """
        Fit the model using X and references.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (Time x Voxels).
        y : Ignored
        references : array-like, shape (n_components, n_features)
            Reference spatial maps (Group ICs).
            REQUIRED.

        Returns
        -------
        self
        """
        X = validation.check_array(X, dtype=np.float64)

        if references is None:
            raise ValueError("references must be provided for GIG-ICA.")

        references = np.asarray(references)
        if references.ndim != 2:
            raise ValueError("references must be 2D array (n_components, n_features).")

        if references.shape[1] != X.shape[1]:
            raise ValueError(
                f"references features ({references.shape[1]}) do not match X features ({X.shape[1]})."
            )

        self.references_ = references

        self.components_ = _algorithm.gig_ica_fit(
            X,
            references,
            alpha=self.alpha,
            whiten=self.whiten,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        return self

    def transform(self, X):
        """
        Apply the signals to X to get time courses.
        Return Time Courses (Time x Comp).
        """
        validation.check_is_fitted(self, ["components_"])
        X = validation.check_array(X, dtype=np.float64)

        # eq 11
        # A = XY^+
        # eq 12 shows an approx
        # T_i = E(XY_i)
        return np.dot(X, self.components_.T) / X.shape[1]
