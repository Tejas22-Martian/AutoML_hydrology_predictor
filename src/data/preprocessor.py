"""
Preprocessing pipeline for streamflow data.

Handles missing values, normalization, temporal splitting, and SMOGN-based
oversampling for extreme event balancing.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger("streamflow_automl.data.preprocessor")


class StreamflowPreprocessor:
    """End-to-end preprocessing for CAMELS-IND streamflow prediction."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.scaler = None
        self.feature_columns = None
        self.test_meta = None
        self._cols_to_drop = []        # high-missing columns, decided on train
        self._feature_medians = None   # per-column fill medians, fit on train only
        self.target_column = self.config.get("target_column", "streamflow_mm_day")

        norm_method = self.config.get("normalization", "standard")
        self.scaler = StandardScaler() if norm_method == "standard" else MinMaxScaler()

    def fit_transform(
        self,
        data: pd.DataFrame,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
    ) -> tuple:
        """
        Full preprocessing: clean, split, normalize, and optionally apply SMOGN.

        Returns (X_train, X_test, y_train, y_test) as numpy arrays.
        """
        # Drop invalid-target rows and split *before* the per-catchment
        # missing-value fill. The groupby/ffill over ~145 columns on the full
        # pooled frame (~1.9M rows) is the memory bottleneck (OOM on a 15 GB box);
        # running it on the bounded train/test frames keeps it tractable.
        data = self._remove_invalid_targets(data)
        train_df, test_df = self._temporal_split(
            data, train_start, train_end, test_start, test_end
        )

        # Cap training rows here, before the fill and SMOGN, so all downstream
        # work is bounded. An even temporal stride preserves train-period coverage.
        max_train_rows = self.config.get("max_train_rows")
        if max_train_rows and len(train_df) > max_train_rows:
            idx = np.linspace(0, len(train_df) - 1, int(max_train_rows)).astype(int)
            train_df = train_df.iloc[idx]
            logger.info(f"Subsampled train rows to {len(train_df)} (cap {max_train_rows})")

        # Missing-value handling: fit the drop list + fill medians on train only
        # (computing them over train+test would leak), reuse them for test.
        train_df = self._handle_missing_values(train_df, fit=True)
        test_df = self._handle_missing_values(test_df, fit=False)

        feature_cols = self._identify_features(train_df)
        self.feature_columns = feature_cols

        # Retain test-set metadata (date + catchment) so downstream plots/reports
        # can align predictions to the correct dates and catchments.
        meta_cols = [c for c in ("date", "catchment_id") if c in test_df.columns]
        self.test_meta = test_df[meta_cols].reset_index(drop=True)

        X_train = train_df[feature_cols].values
        y_train = train_df[self.target_column].values
        X_test = test_df[feature_cols].values
        y_test = test_df[self.target_column].values

        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_test = self.scaler.transform(X_test)

        smogn_config = self.config.get("smogn", {})
        if smogn_config.get("enabled", False):
            X_train, y_train = self._apply_smogn(
                X_train, y_train, feature_cols, smogn_config
            )

        logger.info(
            f"Preprocessing complete: "
            f"train={X_train.shape[0]} samples, test={X_test.shape[0]} samples, "
            f"features={X_train.shape[1]}"
        )
        return X_train, X_test, y_train, y_test

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data with the fitted scaler.

        Used to score catchments outside the training set (e.g. all 242 for risk
        maps). Missing engineered columns are reindexed to the fitted feature set
        (filled with 0 after scaling-safe median handling) so the matrix always
        matches the model's expected inputs.
        """
        if self.scaler is None or self.feature_columns is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        data = self._handle_missing_values(data, fit=False)
        X = data.reindex(columns=self.feature_columns, fill_value=0.0)
        return self.scaler.transform(X.values)

    def _handle_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Fill missing values in feature columns without leaking across catchments.

        The target is never imputed (rows with a missing target are dropped later).
        Feature gaps are forward-filled *within each catchment* (past-only, so a
        gappy streamflow-lag feature cannot leak a future value into a training
        row), then any residual NaNs (e.g. a catchment's leading rows) are filled
        with the column median. On fit the high-missing drop list and fill medians
        are learned from the train frame; on transform they are reused (so test /
        ungauged data never influence the medians).
        """
        data = data.copy()
        protected = {self.target_column, "date", "catchment_id", "basin"}
        max_missing = self.config.get("max_missing_pct", 0.3)

        feature_cols = [c for c in data.columns if c not in protected]
        if fit:
            missing_pct = data[feature_cols].isnull().mean()
            self._cols_to_drop = missing_pct[missing_pct > max_missing].index.tolist()
            if self._cols_to_drop:
                logger.warning(
                    f"Dropping {len(self._cols_to_drop)} feature columns with "
                    f">{max_missing*100:.0f}% missing"
                )
        cols_to_drop = [c for c in self._cols_to_drop if c in data.columns]
        if cols_to_drop:
            data = data.drop(columns=cols_to_drop)
            feature_cols = [c for c in feature_cols if c not in cols_to_drop]

        numeric_cols = [
            c for c in feature_cols
            if c in data.columns and np.issubdtype(data[c].dtype, np.number)
        ]
        if "catchment_id" in data.columns:
            data[numeric_cols] = data.groupby("catchment_id")[numeric_cols].ffill()
        else:
            data[numeric_cols] = data[numeric_cols].ffill()
        if fit:
            self._feature_medians = data[numeric_cols].median()
        data[numeric_cols] = data[numeric_cols].fillna(self._feature_medians)

        remaining = data[numeric_cols].isnull().sum().sum()
        if remaining > 0:
            logger.warning(f"{remaining} feature missing values remain after handling")

        return data

    def _remove_invalid_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where the target variable is invalid."""
        initial_len = len(data)
        data = data.dropna(subset=[self.target_column])
        data = data[data[self.target_column] >= 0]
        removed = initial_len - len(data)
        if removed > 0:
            logger.info(f"Removed {removed} rows with invalid target values")
        return data

    def _temporal_split(
        self,
        data: pd.DataFrame,
        train_start: Optional[str],
        train_end: Optional[str],
        test_start: Optional[str],
        test_end: Optional[str],
    ) -> tuple:
        """Split data temporally (no data leakage from future to past)."""
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])
            data = data.sort_values("date")

        cfg = self.config
        train_start = train_start or cfg.get("train_start")
        train_end = train_end or cfg.get("train_end")
        test_start = test_start or cfg.get("test_start")
        test_end = test_end or cfg.get("test_end")

        if train_end and test_start:
            train_mask = data["date"] <= pd.Timestamp(train_end)
            test_mask = data["date"] >= pd.Timestamp(test_start)
            if train_start:
                train_mask &= data["date"] >= pd.Timestamp(train_start)
            if test_end:
                test_mask &= data["date"] <= pd.Timestamp(test_end)
            return data[train_mask].copy(), data[test_mask].copy()

        split_idx = int(len(data) * 0.8)
        return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()

    def _identify_features(self, data: pd.DataFrame) -> list:
        """Identify feature columns (exclude target, date, IDs)."""
        exclude = {
            self.target_column, "date", "catchment_id", "basin",
            "latitude", "longitude",
        }
        return [
            c for c in data.columns
            if c not in exclude and data[c].dtype in [np.float64, np.float32, np.int64]
        ]

    def _apply_smogn(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        smogn_config: dict,
    ) -> tuple:
        """
        Apply SMOGN (Synthetic Minority Over-sampling for Regression with Gaussian Noise)
        to handle extreme event imbalance.

        Falls back to a manual percentile-based oversampling if smogn package unavailable.
        """
        try:
            import smogn

            df = pd.DataFrame(X, columns=feature_names)
            df["target"] = y

            balanced = smogn.smoter(
                data=df,
                y="target",
                k=smogn_config.get("k_neighbors", 5),
                samp_method="extreme",
                rel_thres=smogn_config.get("relevance_threshold", 0.9),
            )

            X_balanced = balanced[feature_names].values
            y_balanced = balanced["target"].values
            logger.info(
                f"SMOGN applied: {len(y)} -> {len(y_balanced)} samples"
            )
            return X_balanced, y_balanced

        except ImportError:
            logger.warning("smogn package not available, using manual oversampling")
            return self._manual_extreme_oversampling(X, y, smogn_config)

    def _manual_extreme_oversampling(
        self, X: np.ndarray, y: np.ndarray, config: dict
    ) -> tuple:
        """Fallback: oversample extreme values by adding Gaussian noise."""
        threshold_high = np.percentile(y, 95)
        threshold_low = np.percentile(y, 5)

        extreme_mask = (y >= threshold_high) | (y <= threshold_low)
        X_extreme = X[extreme_mask]
        y_extreme = y[extreme_mask]

        if len(X_extreme) == 0:
            return X, y

        rng = np.random.default_rng(42)
        n_synthetic = int(len(X_extreme) * config.get("over_sampling_pct", 0.5))
        indices = rng.choice(len(X_extreme), size=n_synthetic, replace=True)

        noise_scale = 0.05 * np.std(X, axis=0)
        X_synthetic = X_extreme[indices] + rng.normal(0, noise_scale, (n_synthetic, X.shape[1]))
        y_noise_scale = 0.02 * np.std(y)
        y_synthetic = y_extreme[indices] + rng.normal(0, y_noise_scale, n_synthetic)

        X_augmented = np.vstack([X, X_synthetic])
        y_augmented = np.concatenate([y, y_synthetic])

        shuffle_idx = rng.permutation(len(y_augmented))
        logger.info(f"Manual oversampling: {len(y)} -> {len(y_augmented)} samples")
        return X_augmented[shuffle_idx], y_augmented[shuffle_idx]
