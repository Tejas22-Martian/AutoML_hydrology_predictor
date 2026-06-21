"""
Model interpretability using SHAP values and Partial Dependence Plots.

Explains which features drive streamflow predictions and how they influence
flood/drought risk at the catchment level.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("streamflow_automl.models.interpretability")


class SHAPExplainer:
    """SHAP-based model interpretability for streamflow prediction."""

    def __init__(self, model, feature_names: list, output_dir: str = "outputs/figures"):
        self.model = model
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shap_values = None
        self.explainer = None

    def compute_shap_values(
        self, X: np.ndarray, sample_size: Optional[int] = 500
    ) -> np.ndarray:
        """Compute SHAP values for the given data."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available, using permutation importance fallback")
            return self._permutation_importance(X)

        if sample_size and len(X) > sample_size:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(X), size=sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        model_type = type(self.model).__name__
        if model_type in ("RandomForestRegressor", "XGBRegressor", "LGBMRegressor"):
            self.explainer = shap.TreeExplainer(self.model)
        elif model_type == "LSTMRegressor":
            # KernelExplainer would call predict() thousands of times and the LSTM
            # rebuilds sequences each call - infeasible. Use permutation importance.
            logger.info("LSTM model: using permutation importance instead of SHAP")
            return self._permutation_importance(X_sample[:150])
        else:
            # KernelExplainer is expensive: keep the sample and background small.
            X_sample = X_sample[:150]
            background = shap.sample(X_sample, min(50, len(X_sample)))
            self.explainer = shap.KernelExplainer(self.model.predict, background)

        self.shap_values = self.explainer.shap_values(X_sample)
        logger.info(f"SHAP values computed for {len(X_sample)} samples")
        return self.shap_values

    def plot_summary(self, X: np.ndarray, save: bool = True) -> None:
        """Generate SHAP summary plot (beeswarm)."""
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not available for plotting")
            return

        if self.shap_values is None:
            self.compute_shap_values(X)

        sample_size = len(self.shap_values)
        X_plot = X[:sample_size] if len(X) >= sample_size else X

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, X_plot,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title("SHAP Feature Importance - Streamflow Prediction")
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "shap_summary.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"SHAP summary plot saved to {filepath}")
        plt.close()

    def plot_feature_importance(self, top_n: int = 20, save: bool = True) -> pd.DataFrame:
        """Plot mean absolute SHAP values as bar chart."""
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": importance,
        }).sort_values("mean_abs_shap", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        top = importance_df.head(top_n)
        ax.barh(range(len(top)), top["mean_abs_shap"].values, color="#2196F3")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title(f"Top {top_n} Feature Importances (SHAP)")
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "shap_feature_importance.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {filepath}")
        plt.close()

        return importance_df

    def plot_dependence(
        self, feature_name: str, X: np.ndarray, save: bool = True
    ) -> None:
        """Generate SHAP dependence plot for a single feature."""
        try:
            import shap
        except ImportError:
            return

        if self.shap_values is None:
            self.compute_shap_values(X)

        if feature_name not in self.feature_names:
            logger.warning(f"Feature '{feature_name}' not found")
            return

        sample_size = len(self.shap_values)
        X_plot = X[:sample_size]

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feature_name, self.shap_values, X_plot,
            feature_names=self.feature_names,
            show=False, ax=ax,
        )
        plt.title(f"SHAP Dependence: {feature_name}")
        plt.tight_layout()

        if save:
            safe_name = feature_name.replace("/", "_")
            filepath = self.output_dir / f"shap_dependence_{safe_name}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_partial_dependence(
        self, X: np.ndarray, features: list, save: bool = True
    ) -> None:
        """Generate Partial Dependence Plots for specified features."""
        from sklearn.inspection import PartialDependenceDisplay

        feature_indices = [
            self.feature_names.index(f) for f in features
            if f in self.feature_names
        ]

        if not feature_indices:
            logger.warning("No valid features for PDP")
            return

        fig, axes = plt.subplots(
            nrows=2, ncols=(len(feature_indices) + 1) // 2,
            figsize=(5 * ((len(feature_indices) + 1) // 2), 8),
        )

        PartialDependenceDisplay.from_estimator(
            self.model, X, feature_indices,
            feature_names=self.feature_names,
            ax=axes.ravel()[:len(feature_indices)],
        )

        plt.suptitle("Partial Dependence Plots", fontsize=14)
        plt.tight_layout()

        if save:
            filepath = self.output_dir / "partial_dependence_plots.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            logger.info(f"PDP saved to {filepath}")
        plt.close()

    def _permutation_importance(self, X: np.ndarray) -> np.ndarray:
        """Fallback: estimate feature importance via output variance per feature."""
        base_pred = self.model.predict(X)
        importances = np.zeros(X.shape[1])
        rng = np.random.default_rng(42)

        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X_permuted[:, i])
            perm_pred = self.model.predict(X_permuted)
            importances[i] = np.mean((base_pred - perm_pred) ** 2)

        fake_shap = np.zeros_like(X)
        for i in range(X.shape[1]):
            fake_shap[:, i] = importances[i]

        self.shap_values = fake_shap
        return fake_shap
