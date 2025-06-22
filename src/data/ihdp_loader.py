import os
import numpy as np
from pathlib import Path
from typing import Tuple
import urllib.request

# Constants for filenames and URLs of IHDP benchmark datasets
TRAIN_DATASET = "ihdp_npci_1-100.train.npz"
TEST_DATASET = "ihdp_npci_1-100.test.npz"
TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


def _download_if_needed(file_path: Path, url: str):
    """
    Downloads a dataset from a given URL if it doesn't already exist locally.

    Args:
        file_path (Path): Local path to store the file.
        url (str): URL to download the file from.
    """
    if not file_path.exists():
        print(f"📥 Downloading {file_path.name}...")
        urllib.request.urlretrieve(url, file_path)


def _load_npz(file: Path, get_po: bool = True):
    """
    Loads .npz file and extracts relevant arrays.

    Args:
        file (Path): Path to the .npz file.
        get_po (bool): Whether to extract true potential outcomes.

    Returns:
        dict: Dictionary containing features (X), treatment (w), factual outcomes (y),
              counterfactual outcomes (ycf), and true mu0/mu1 (if available).
    """
    data = np.load(file)
    output = {
        "X": data["x"],         # Features: shape (n, d, 100)
        "w": data["t"],         # Treatment assignments: shape (n, 100)
        "y": data["yf"],        # Factual outcomes: shape (n, 100)
        "ycf": data.get("ycf", None),  # Counterfactuals, may be missing
        "mu0": data.get("mu0", None) if get_po else None,  # True potential outcome Y(0)
        "mu1": data.get("mu1", None) if get_po else None,  # True potential outcome Y(1)
    }
    output["HAVE_TRUTH"] = output["ycf"] is not None
    return output


def get_ihdp_experiment(
    data_path: Path,
    exp_idx: int = 1,
    rescale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads a single replication of the IHDP benchmark dataset for CMGP evaluation. (adapted from https://github.com/clinicalml/cfrnet)

    Args:
        data_path (Path): Directory where the .npz files are or will be stored.
        exp_idx (int): Index of the replication (1 to 100).
        rescale (bool): Whether to normalize outcomes by the std of the CATE.

    Returns:
        Tuple of arrays:
            - X_train: Covariates (n_train, d)
            - T_train: Treatment assignment (n_train,)
            - Y_train: Observed outcome (n_train,)
            - pot_y_train: True potential outcomes [Y0, Y1] (n_train, 2)
            - X_test: Covariates (n_test, d)
            - T_test: Treatment assignment (n_test,)
            - Y_test: Observed outcome (n_test,)
            - pot_y_test: True potential outcomes [Y0, Y1] (n_test, 2)
    """
    # Download IHDP dataset files if not present
    train_path = data_path / TRAIN_DATASET
    test_path = data_path / TEST_DATASET

    _download_if_needed(train_path, TRAIN_URL)
    _download_if_needed(test_path, TEST_URL)

    # Load full .npz files (contains 100 experiments)
    train = _load_npz(train_path)
    test = _load_npz(test_path)

    # Extract the specified experiment (slice over last axis)
    def slice_exp(D):
        return {
            "X": D["X"][:, :, exp_idx - 1],
            "w": D["w"][:, exp_idx - 1],
            "y": D["y"][:, exp_idx - 1],
            "mu0": D["mu0"][:, exp_idx - 1],
            "mu1": D["mu1"][:, exp_idx - 1],
        }

    train = slice_exp(train)
    test = slice_exp(test)

    # Optional rescaling: normalize mu0/mu1 by CATE std (if std > 1)
    if rescale:
        cate = train["mu1"] - train["mu0"]
        sd = np.std(cate)
        if sd > 1:
            error = train["y"] - train["w"] * train["mu1"] - (1 - train["w"]) * train["mu0"]
            train["mu0"] /= sd
            train["mu1"] /= sd
            train["y"] = train["w"] * train["mu1"] + (1 - train["w"]) * train["mu0"] + error

            test["mu0"] /= sd
            test["mu1"] /= sd

    # Stack potential outcomes into 2-column format: [Y0, Y1]
    po_train = np.stack([train["mu0"], train["mu1"]], axis=1)
    po_test = np.stack([test["mu0"], test["mu1"]], axis=1)

    return (
        train["X"], train["w"], train["y"], po_train,
        test["X"], test["w"], test["y"], po_test
    )
