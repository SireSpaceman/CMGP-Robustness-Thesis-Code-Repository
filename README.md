# CMGP Enhancements for Robust Causal Inference

This repository contains code and Jupyter notebooks for evaluating two enhancements to the Causal Multi-task Gaussian Process (CMGP) model for individualized treatment effect (ITE) estimation. The modifications aim to improve model robustness in high-dimensional, imbalanced, or low-sample regimes, without degrading performance in well-behaved settings.

## Abstract

Gaussian Processes (GPs) offer a flexible framework for causal inference, but standard multi-task GPs struggle in regimes with limited overlap, high dimensionality, and small sample sizes. This work investigates two enhancements to the Causal Multi-task Gaussian Process (CMGP) model: variance-weighted ARD regularization and overlap-aware kernel scaling. These modifications introduce data-aware priors and localized regularization to improve generalization under adverse conditions. Synthetic and semi-synthetic experiments demonstrate that the enhancements offer robustness in imbalanced or noisy settings without harming performance when structural assumptions hold.

## Project Structure

All experimental notebooks are located in the root directory:

- `variance_ard_enhancment_exp.ipynb`: Experiments with fixed confounders and increasing effect modifiers (Failure Mode 1 – Dimensionality).
- `variance_ard_enhancment_exp_conf.ipynb`: Experiments with fixed modifiers and increasing confounders (Failure Mode 1 – Confounding).
- `failure_mode_dimensionality.ipynb`: Baseline CMGP behavior under dimensionality stress.
- `failure_mode_dimensionality_conf.ipynb`: Baseline CMGP behavior under increasing confounding.
- `failure_mode_overlap.ipynb`: Overlap-aware kernel evaluation under treatment imbalance (Failure Mode 2 – Overlap).
- `idhp_benchmark.ipynb`: Evaluation on the IHDP semi-synthetic benchmark with real covariates and simulated outcomes.

## Getting Started

To run the notebooks locally:

1. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
2. Install dependencies
   ```bash
   pip install -r requirements.txt
