"""
A data-generating process (DGP) of an observational dataset
where multiple variables are measured, some of them confounders,
but others are not. Extended from the original implementation by R.K.A. Karlsson
to support treatment imbalance control and variance normalization.

Author: R.K.A. Karlsson (original), Adapted and extended by Logan Ritter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialDGP:
    """
    Simulates observational data with user-specified confounders, modifiers, and instruments.
    Supports both continuous and binary outcomes, polynomial outcome relationships, and optional
    treatment imbalance via the `treatment_ratio` parameter.

    Attributes:
        - polynomial_degree: Degree of outcome polynomial (nonlinear relationships).
        - confounding_strength: Scalar multiplier on confounder features.
        - n_effect_modifiers: # of variables that modulate treatment effects.
        - n_confounders: # of shared predictors of treatment and outcome.
        - n_instruments: # of predictors of treatment only.
        - n_noise: # of noise variables.
        - binary_outcome: If True, outcomes are Bernoulli samples.
        - treatment_ratio: Optional treatment/control split ratio (imbalance).
        - seed: RNG seed.
    """

    def __init__(
        self,
        polynomial_degree: int = 2,
        confounding_strength: float = 1.0,
        n_effect_modifiers: int = 1,
        n_confounders: int = 1,
        n_instruments: int = 1,
        n_noise: int = 0,
        binary_outcome: bool = False,
        treatment_ratio: float = None,  # ➕ NEW: Allows imbalance in treatment assignment
        seed: int = 42,
    ):
        self.degree = polynomial_degree
        self.confounding_strength = confounding_strength
        self.n_effect_modifiers = n_effect_modifiers
        self.n_confounders = n_confounders
        self.n_instruments = n_instruments
        self.n_noise = n_noise
        self.binary_outcome = binary_outcome
        self.treatment_ratio = treatment_ratio  # ➕ NEW
        self.random_state = np.random.RandomState(seed)

        # Number of features for treatment and outcome generation
        self.n_treat_features = n_confounders + n_instruments
        self.n_outcome_features = n_confounders + n_effect_modifiers
        if self.n_outcome_features < 1:
            raise ValueError("Outcome model must have at least one covariate.")

        # Prepare polynomial basis for outcome surface
        self.poly_outcome = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.poly_outcome.fit(np.zeros((1, self.n_outcome_features)))
        self.n_poly_outcome = self.poly_outcome.transform(
            np.zeros((1, self.n_outcome_features))
        ).shape[1]

        # Coefficients for baseline and treatment effect surfaces
        self.baseline_coefs = self.random_state.normal(0, 1, size=self.n_poly_outcome)
        self.tau_coefs = self.random_state.normal(0, 1, size=self.n_poly_outcome)

        # Treatment assignment weights
        if self.n_treat_features < 1:
            raise ValueError("Treatment model must have at least one covariate.")
        self.treat_coefs = self.random_state.normal(0, 1, size=self.n_treat_features)

    def get_feature_names(self) -> list:
        """
        Returns:
            List[str]: Names of all covariates, grouped by role.
        """
        return (
            [f"X_c{i+1}" for i in range(self.n_confounders)] +
            [f"X_i{i+1}" for i in range(self.n_instruments)] +
            [f"X_m{i+1}" for i in range(self.n_effect_modifiers)] +
            [f"X_n{i+1}" for i in range(self.n_noise)]
        )

    def sample(self, n: int) -> pd.DataFrame:
        """
        Samples a synthetic observational dataset with potential outcomes.

        Args:
            n (int): Number of samples.

        Returns:
            pd.DataFrame: Data with observed features, treatment, outcome,
                          and counterfactual outcomes (_Y0, _Y1, _cate).
        """
        # Generate feature blocks by role
        X_c = self._gen_covariate_block(n, self.n_confounders) * self.confounding_strength
        X_i = self._gen_covariate_block(n, self.n_instruments)
        X_m = self._gen_covariate_block(n, self.n_effect_modifiers)
        X_n = self._gen_covariate_block(n, self.n_noise)

        # Treatment assignment logic
        X_treat = self._safe_hstack([X_c, X_i])
        treat_logit = X_treat @ self.treat_coefs

        if self.treatment_ratio is not None:
            # ➕ NEW: Imposes imbalance via log-odds bias
            bias = np.log(self.treatment_ratio / (1 - self.treatment_ratio))
            prob = self._sigmoid(treat_logit + bias)
            T = self.random_state.binomial(1, p=prob)
        else:
            T = self.random_state.binomial(1, p=self._sigmoid(treat_logit))

        # Outcome generation
        X_outcome = np.hstack([X_c, X_m])
        X_poly_outcome = self.poly_outcome.transform(X_outcome)

        baseline = X_poly_outcome @ self.baseline_coefs
        cate = X_poly_outcome @ self.tau_coefs

        Y0_cont = baseline + self.random_state.normal(0, 1, size=n)
        Y1_cont = Y0_cont + cate

        # Binary vs continuous outcomes
        if self.binary_outcome:
            Y0 = self.random_state.binomial(1, self._sigmoid(Y0_cont))
            Y1 = self.random_state.binomial(1, self._sigmoid(Y1_cont))
        else:
            Y0 = Y0_cont
            Y1 = Y1_cont

        # ➕ NEW: Rescale CATEs if too large for numerical stability
        sd_cate = np.std(cate)
        if sd_cate > 1:
            Y0 /= sd_cate
            Y1 /= sd_cate
            cate /= sd_cate

        Y = T * Y1 + (1 - T) * Y0

        # Combine into final DataFrame
        X_full = self._safe_hstack([X_c, X_i, X_m, X_n])
        data = pd.DataFrame(X_full, columns=self.get_feature_names())
        data["T"] = T
        data["Y"] = Y
        data["_Y0"] = Y0
        data["_Y1"] = Y1
        data["_cate"] = cate

        return data

    def _gen_covariate_block(self, n, d):
        """Helper to generate a block of normal covariates (n x d)."""
        return self.random_state.normal(0, 1, size=(n, d)) if d > 0 else np.zeros((n, 0))

    def _safe_hstack(self, arrays):
        """Stacks arrays horizontally, omitting empty blocks."""
        arrays = [a for a in arrays if a.shape[1] > 0]
        return np.hstack(arrays) if arrays else np.zeros((1, 0))

    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return 1 / (1 + np.exp(-x))

