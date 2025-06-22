"""
evaluate.py

Evaluation module for CMGP (Causal Multi-task Gaussian Processes).
Trains a CMGP model on a given dataset and computes relevant causal inference metrics.

Uses:
- src.load.load_dataset: for data ingestion
- src.cmgp.cmgp.CMGP: core model for counterfactual prediction
- src.cmgp.utils.metrics: √PEHE, ATE, ATT, RPol, and confidence intervals

Author: Adapted by Logan Ritter, 2025
"""

import numpy as np
from pathlib import Path

from src.load import load_dataset
from src.cmgp.cmgp import CMGP
from src.cmgp.utils.metrics import (
    sqrt_PEHE_with_diff,
    ATT,
    RPol,
    mean_confidence_interval
)


def evaluate_dataset(dataset_name: str, data_path: Path, **kwargs) -> dict:
    """
    Run a full evaluation pipeline on a CMGP model trained on the given dataset.

    Parameters:
        dataset_name (str): 'ihdp' or 'synthetic'
        data_path (Path): Path to the dataset folder
        **kwargs:
            Passed to load_dataset and CMGP, e.g.,
                - exp_idx: (int) IHDP experiment index
                - seed: (int) for reproducibility
                - max_gp_iterations: (int) max optimization steps for CMGP
                - overlap_scaling: (bool) enable hybrid overlap-aware kernel scaling
                - variance_ard: (bool) apply ARD lengthscale scaling via ridge-T learner

    Returns:
        dict: A dictionary of evaluation metrics:
            {
                'Dataset', '√PEHE', 'Mean ITE CI', 'ATE', 'ATT', 'RPol'
            }
    """
    data = load_dataset(dataset_name, data_path, **kwargs)

    X_train = data["X_train"]
    T_train = data["T_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    T_test = data["T_test"]
    Y_test = data["Y_test"]
    Y0_test = data["pot_y_test"][:, 0]
    Y1_test = data["pot_y_test"][:, 1]

    model = CMGP(
        X_train, T_train, Y_train,
        max_gp_iterations=kwargs.get("max_gp_iterations", 100),
        overlap_scaling=kwargs.get("overlap_scaling", False),
        variance_ard=kwargs.get("variance_ard", False)
    )
    ITE_pred = model.predict(X_test).flatten()
    ITE_true = Y1_test - Y0_test
    Y0Y1_test = np.column_stack([Y0_test, Y1_test])

    pehe = sqrt_PEHE_with_diff(Y0Y1_test, ITE_pred)
    ci_mean, ci_width = mean_confidence_interval(ITE_pred)
    ate = np.mean(ITE_pred) - np.mean(ITE_true)
    att = ATT(T_test, Y_test, Y0Y1_test)
    rpol = RPol(T_test, Y_test, Y0Y1_test)

    return {
        "Dataset": dataset_name,
        "√PEHE": round(pehe, 4),
        "Mean ITE CI": (round(ci_mean, 4), round(ci_width, 4)),
        "ATE": round(ate, 4),
        "ATT": round(att, 4),
        "RPol": round(rpol, 4)
    }
