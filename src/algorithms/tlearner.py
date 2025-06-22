"""
A T-learner for estimating the CATE, can act as a simple baseline.

Author: R.K.A. Karlsson
"""

from sklearn.base import clone, BaseEstimator
import numpy as np


class TLearner:
    def __init__(self, base_model: BaseEstimator):
        self.base_model_0 = clone(base_model)
        self.base_model_1 = clone(base_model)
        self.model_0 = None
        self.model_1 = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        """
        X: (n_samples, n_features) covariates
        T: (n_samples,) binary treatment indicator
        Y: (n_samples,) observed outcome
        """
        X0, Y0 = X[T == 0], Y[T == 0]
        X1, Y1 = X[T == 1], Y[T == 1]

        self.model_0 = clone(self.base_model_0).fit(X0, Y0)
        self.model_1 = clone(self.base_model_1).fit(X1, Y1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Estimates treatment effect for each row in X.

        Returns:
            tau_hat: (n_samples,) estimated individual treatment effects
        """
        mu0 = self.model_0.predict(X)
        mu1 = self.model_1.predict(X)
        return mu1 - mu0
