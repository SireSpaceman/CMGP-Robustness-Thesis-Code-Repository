"""
load.py

This module provides standardized access to datasets for causal inference experiments.
It supports both:
- The real-world IHDP dataset (automatically downloaded if not found)
- Synthetic data generated via a flexible Polynomial DGP

It outputs data in a consistent dictionary format suitable for CMGP model training and evaluation.

Available datasets:
- 'ihdp': Loads a specific simulation from the IHDP dataset (based on experiment index)
- 'synthetic': Uses a controlled Polynomial Data-Generating Process to simulate outcomes

Author: Adapted by Logan Ritter, 2025
"""

import numpy as np
from pathlib import Path


def load_dataset(name: str, data_path: Path, **kwargs) -> dict:
    """
    Load and prepare a dataset in a consistent format for model consumption.

    Parameters:
        name (str): Dataset name ('ihdp' or 'synthetic')
        data_path (Path): Path where data is stored (or will be downloaded to)
        **kwargs:
            For 'ihdp':
                - exp_idx (int): Experiment index to select (1–100)
            For 'synthetic':
                - seed (int): Random seed for reproducibility
                - polynomial_degree (int): Degree of polynomial for outcome model
                - n_confounders (int): Number of confounders
                - n_effect_modifiers (int): Effect-modifying covariates
                - n_instruments (int): Instrumental variables
                - confounding_strength (float): Strength of confounding in treatment assignment
                - treatment_ratio (float or None): Fixed treatment ratio (overrides random assignment)
                - n_samples (int): Total number of samples to generate
                - test_size (float): Proportion of data used for testing

    Returns:
        dict: {
            "X_train", "T_train", "Y_train", "pot_y_train",
            "X_test", "T_test", "Y_test", "pot_y_test",
            "feature_names", "dgp" (only for synthetic)
        }
    """
    if name.lower() == "ihdp":
        from src.data.ihdp_loader import get_ihdp_experiment

        X_train, T_train, Y_train, pot_y_train, X_test, T_test, Y_test, pot_y_test = get_ihdp_experiment(
            data_path, exp_idx=kwargs.get("exp_idx", 1), rescale=True
        )
        feature_names = [f"X{i}" for i in range(X_train.shape[1])]

        return {
            "X_train": X_train,
            "T_train": T_train,
            "Y_train": Y_train,
            "pot_y_train": pot_y_train,
            "X_test": X_test,
            "T_test": T_test,
            "Y_test": Y_test,
            "pot_y_test": pot_y_test,
            "feature_names": feature_names,
            "dgp": None  # Real data has no DGP object
        }

    elif name.lower() == "synthetic":
        from src.data.polynomial_dgp import PolynomialDGP
        from sklearn.model_selection import train_test_split

        dgp = PolynomialDGP(
            seed=kwargs.get("seed", 42),
            polynomial_degree=kwargs.get("polynomial_degree", 2),
            n_confounders=kwargs.get("n_confounders", 3),
            n_effect_modifiers=kwargs.get("n_effect_modifiers", 3),
            n_instruments=kwargs.get("n_instruments", 0),
            confounding_strength=kwargs.get("confounding_strength", 1.0),
            treatment_ratio=kwargs.get("treatment_ratio", None)
        )

        df = dgp.sample(n=kwargs.get("n_samples", 1000))
        feature_names = dgp.get_feature_names()

        X = df[feature_names].values
        T = df["T"].values
        Y = df["Y"].values
        Y0 = df["_Y0"].values
        Y1 = df["_Y1"].values
        pot_y = np.column_stack([Y0, Y1])

        X_train, X_test, T_train, T_test, Y_train, Y_test, pot_y_train, pot_y_test = train_test_split(
            X, T, Y, pot_y, test_size=kwargs.get("test_size", 0.2), random_state=kwargs.get("seed", 42)
        )

        return {
            "X_train": X_train,
            "T_train": T_train,
            "Y_train": Y_train,
            "pot_y_train": pot_y_train,
            "X_test": X_test,
            "T_test": T_test,
            "Y_test": Y_test,
            "pot_y_test": pot_y_test,
            "feature_names": feature_names,
            "dgp": dgp  # Helpful for introspecting or regenerating synthetic data
        }

    else:
        raise ValueError(f"Unsupported dataset name: {name}")
