# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
# Original source: https://github.com/AMLab-Amsterdam/CMGP
# Paper: Ahmed M. Alaa and Mihaela van der Schaar,
#        "Bayesian Inference of Individualized Treatment Effects using Multi-task Gaussian Processes", NeurIPS 2017.
#
# Modifications by Logan Ritter (2025):
# - Added overlap-aware kernel scaling based on hybrid local variance and kNN overlap
# - Added variance-based ARD initialization using a Ridge T-learner
# - Introduced two flags: `overlap_scaling` and `variance_ard` to control these mechanisms
# - Integrated enhancements into CMGP hyperparameter initialization

import GPy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class CMGP:
    """
        Initialize the CMGP model with optional enhancements.

            Parameters:
            - X: Covariate matrix (n x d)
            - Treatments: Binary treatment assignments (n,)
            - Y: Observed outcomes (n,)
            - mode: "CMGP" or "NSGP" kernel setup
            - max_gp_iterations: Maximum BFGS optimization steps
            - overlap_scaling: Apply overlap-aware per-arm lengthscale scaling (new)
            - variance_ard: Apply ARD initialization based on conditional CATE variance (new)
    """
    def __init__(
            self,
            X: np.ndarray,
            Treatments: np.ndarray,
            Y: np.ndarray,
            mode: str = "CMGP",
            max_gp_iterations: int = 1000,
            overlap_scaling: bool = False,
            variance_ard: bool = False,
    ) -> None:
        X = np.asarray(X)
        Y = np.asarray(Y)
        Treatments = np.asarray(Treatments)

        self.dim = X.shape[1]
        self.dim_outcome = len(np.unique(Y))
        self.mode = mode
        self.max_gp_iterations = max_gp_iterations
        self.overlap_scaling = overlap_scaling
        self.variance_ard = variance_ard

        if self.dim < 1 or not isinstance(self.dim, int):
            raise ValueError("Input dimension must be a positive integer.")

        self._fit(X, Treatments, Y)

    def _fit(self, Train_X, Train_T, Train_Y):
        """
                Predict the individual treatment effect (ITE) for given covariates.

                Returns:
                - A vector of ITE estimates: E[Y|X,T=1] - E[Y|X,T=0]
        """
        Dataset = pd.DataFrame(Train_X)
        Dataset["Y"] = Train_Y
        Dataset["T"] = Train_T

        Feature_names = list(range(self.dim))

        if self.overlap_scaling:
            self._compute_overlap_weights(Train_X, Train_T)

        Dataset0 = Dataset[Dataset["T"] == 0].copy()
        Dataset1 = Dataset[Dataset["T"] == 1].copy()

        X0 = Dataset0[Feature_names].to_numpy()
        y0 = Dataset0["Y"].to_numpy().reshape(-1, 1)
        X1 = Dataset1[Feature_names].to_numpy()
        y1 = Dataset1["Y"].to_numpy().reshape(-1, 1)

        K0 = GPy.kern.RBF(self.dim, ARD=True)
        K1 = GPy.kern.RBF(self.dim, ARD=True)

        kernel_dict = {
            "CMGP": GPy.util.multioutput.LCM(
                input_dim=self.dim, num_outputs=self.dim_outcome, kernels_list=[K0, K1]
            ),
            "NSGP": GPy.util.multioutput.ICM(
                input_dim=self.dim, num_outputs=self.dim_outcome, kernel=K0
            ),
        }

        self.model = GPy.models.GPCoregionalizedRegression(
            X_list=[X0, X1], Y_list=[y0, y1], kernel=kernel_dict[self.mode]
        )

        self._initialize_hyperparameters(Train_X, Train_T, Train_Y)

        try:
            self.model.optimize("bfgs", max_iters=self.max_gp_iterations)
        except np.linalg.LinAlgError as err:
            print("Covariance matrix not invertible. ", err)
            raise err

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_0 = np.hstack([X, np.zeros((X.shape[0], 1))])
        X_1 = np.hstack([X, np.ones((X.shape[0], 1))])

        noise_dict_0 = {"output_index": X_0[:, -1].astype(int).reshape(-1, 1)}
        noise_dict_1 = {"output_index": X_1[:, -1].astype(int).reshape(-1, 1)}

        Y_est_0 = self.model.predict(X_0, Y_metadata=noise_dict_0)[0]
        Y_est_1 = self.model.predict(X_1, Y_metadata=noise_dict_1)[0]

        return Y_est_1 - Y_est_0

    def _initialize_hyperparameters(self, X, T, Y):
        """
                Estimate initial hyperparameters for the CMGP model using KNN smoothers.
                Applies enhanced scaling methods if enabled:
                - `overlap_scaling` scales kernel lengthscales per treatment arm
                - `variance_ard` initializes ARD lengthscales based on conditional CATE variance
        """
        Dataset = pd.DataFrame(X)
        Dataset["Y"] = Y
        Dataset["T"] = T

        Feature_names = list(range(self.dim))

        neigh0 = KNeighborsRegressor(n_neighbors=10).fit(Dataset[Dataset["T"] == 0][Feature_names], Dataset[Dataset["T"] == 0]["Y"])
        neigh1 = KNeighborsRegressor(n_neighbors=10).fit(Dataset[Dataset["T"] == 1][Feature_names], Dataset[Dataset["T"] == 1]["Y"])

        Dataset["Yk0"] = neigh0.predict(Dataset[Feature_names])
        Dataset["Yk1"] = neigh1.predict(Dataset[Feature_names])

        Dataset0 = Dataset[Dataset["T"] == 0].copy()
        Dataset1 = Dataset[Dataset["T"] == 1].copy()

        a0 = np.std(Dataset0["Y"])
        a1 = np.std(Dataset1["Y"])
        b0 = np.cov(Dataset["Yk0"], Dataset["Yk1"])[0, 1] / (a0 * a1)
        b1 = b0
        s0 = np.std(Dataset0["Y"] - Dataset0["Yk0"]) / a0
        s1 = np.std(Dataset1["Y"] - Dataset1["Yk1"]) / a1

        self.model.sum.ICM0.rbf.lengthscale = 10 * np.ones(self.dim)
        self.model.sum.ICM1.rbf.lengthscale = 10 * np.ones(self.dim)

        if self.overlap_scaling:
            self._compute_overlap_weights(X, T)

        if self.overlap_scaling and hasattr(self, 'covariate_scales'):
            self.model.sum.ICM0.rbf.lengthscale *= self.covariate_scales[0]
            self.model.sum.ICM1.rbf.lengthscale *= self.covariate_scales[1]
            print(f"[Covariate Scaling] Applied per-dim scaling to ICM0 and ICM1 kernels")

        if self.variance_ard:
            self.update_ard_lengthscales(X, T, Y)

        self.model.sum.ICM0.rbf.variance = 1
        self.model.sum.ICM1.rbf.variance = 1
        self.model.sum.ICM0.B.W[0] = b0
        self.model.sum.ICM0.B.W[1] = b0
        self.model.sum.ICM1.B.W[0] = b1
        self.model.sum.ICM1.B.W[1] = b1
        self.model.sum.ICM0.B.kappa[0] = a0 ** 2
        self.model.sum.ICM0.B.kappa[1] = 1e-4
        self.model.sum.ICM1.B.kappa[0] = 1e-4
        self.model.sum.ICM1.B.kappa[1] = a1 ** 2

        self.model.mixed_noise.Gaussian_noise_0.variance = s0 ** 2
        self.model.mixed_noise.Gaussian_noise_1.variance = s1 ** 2


    def update_ard_lengthscales(self, X, T, Y, bins=40, alpha=0.7, epsilon=1e-4):
        """
        Initializes ARD kernel lengthscales using conditional variance of predicted CATEs.

        Procedure:
        - Train a Ridge T-learner to estimate ITE
        - Discretize each feature into quantile bins
        - Compute per-feature variance of binned ITE means
        - Inverse-variance scaling is applied to ARD lengthscales

        Parameters:
        - bins: Number of quantile bins per feature
        - alpha: Shrinkage toward uniform scaling (1.0 = full scaling, 0 = no scaling)
        - epsilon: Smoothing factor to avoid division by zero
        """

        X = np.asarray(X)
        T = np.asarray(T)
        Y = np.asarray(Y)
        d = X.shape[1]

        # Standardize features for ridge regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit T-learner
        ridge0 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5).fit(X_scaled[T == 0], Y[T == 0])
        ridge1 = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], cv=5).fit(X_scaled[T == 1], Y[T == 1])
        ITE_pred = ridge1.predict(X_scaled) - ridge0.predict(X_scaled)

        # Estimate per-feature conditional variance of ITE
        df = pd.DataFrame(X_scaled, columns=[f"x{j}" for j in range(d)])
        df["ITE"] = ITE_pred
        feature_var = np.zeros(d)

        for j in range(d):
            try:
                df["bin"] = pd.qcut(df[f"x{j}"], q=bins, labels=False, duplicates="drop")
                grouped = df.groupby("bin")["ITE"].mean()
                feature_var[j] = np.var(grouped.values) if len(grouped) > 1 else 0.0
            except ValueError:
                feature_var[j] = 0.0

        # Inverse-variance scaling with log smoothing
        scaling = 1.0 / (feature_var + epsilon)
        scaling /= np.mean(scaling)  # Normalize
        scaling = alpha * scaling + (1 - alpha)  # Shrinkage

        # Apply to lengthscales
        self.model.sum.ICM0.rbf.lengthscale[:] *= scaling
        self.model.sum.ICM1.rbf.lengthscale[:] *= scaling

        print(f"[Ridge T-Learner ARD Init] First 10 lengthscale scalings:", np.round(scaling[:10], 3))

    def _compute_overlap_weights(self, X, T, k: int = 10, epsilon: float = 1e-4):
        """
        Computes per-arm scaling factors for ARD kernels using a hybrid score.

        Steps:
        1. Estimate joint overlap weights (using kNN treatment density)
        2. Compute local variance for each feature individually
        3. Multiply to form a hybrid matrix: joint_weight × local variance
        4. Aggregate feature-wise means per treatment group
        5. Normalize to obtain scale vectors s^0, s^1 with s^0_j + s^1_j = 2

        Sets:
        - self.covariate_scales: (scales_control, scales_treated)
        These are later applied to the respective RBF kernel lengthscales.
        """
        n, d = X.shape

        # ------------- STEP 1: joint overlap (full space) --------------------
        nbrs_full = NearestNeighbors(n_neighbors=k).fit(X)
        joint_weight = np.empty(n)
        for i in range(n):
            idx = nbrs_full.kneighbors(X[i].reshape(1, -1),
                                       return_distance=False)[0]
            p_i = np.clip(T[idx].mean(), epsilon, 1 - epsilon)
            joint_weight[i] = np.log1p(1.0 / (p_i * (1 - p_i) + epsilon))

        # ------------- STEP 2: local 1-D variances ---------------------------
        local_var = np.zeros((n, d))
        for j in range(d):
            nbrs_1d = NearestNeighbors(n_neighbors=k).fit(X[:, [j]])
            for i in range(n):
                idx = nbrs_1d.kneighbors(X[i, j].reshape(1, -1),
                                         return_distance=False)[0]
                v_ij = X[idx, j].var()
                if v_ij == 0:  # guard against zero variance
                    v_ij = 1.0
                local_var[i, j] = v_ij

        # ------------- STEP 3: hybrid score per unit × feature --------------
        hybrid = joint_weight[:, None] * local_var  # shape (n, d)

        # ------------- STEP 4: group means (per feature) ---------------------
        mask_ctrl = (T == 0)
        mask_trt = ~mask_ctrl
        scales_0 = hybrid[mask_ctrl].mean(axis=0)  # s_j^0
        scales_1 = hybrid[mask_trt].mean(axis=0)  # s_j^1

        # ------------- STEP 5: per-feature normalisation ---------------------
        mean_per_feature = 0.5 * (scales_0 + scales_1)
        mean_per_feature = np.maximum(mean_per_feature, epsilon)
        scales_0 /= mean_per_feature  # now s0j + s1j = 2
        scales_1 /= mean_per_feature

        self.covariate_scales = (scales_0, scales_1)

        # ----------- logging -------------------------------------------------
        print(f"[Hybrid Scaling] scalar ctrl={np.mean(scales_0):.3f}, "
              f"scalar treat={np.mean(scales_1):.3f}")
        print(f"First 10 per-feature ctrl ratios (debug): "
              f"{np.round(scales_0[:10], 2)}")
        print(f"First 10 per-feature treat ratios (debug): "
              f"{np.round(scales_1[:10], 2)}")


