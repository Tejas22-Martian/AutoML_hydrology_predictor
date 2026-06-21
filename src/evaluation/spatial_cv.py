"""
Spatial cross-validation: Prediction in Ungauged Basins (PUB).

WHY THIS IS A STRONG ADDITION
-----------------------------
There are two very different questions you can ask of a streamflow model:

  1. TEMPORAL test: "trained on 1990-2010 for this catchment, can it predict
     2011-2020 for the SAME catchment?"  (what the base project already does)

  2. SPATIAL test (PUB): "trained on catchments A,B,C..., can it predict an
     UNSEEN catchment Z that has no streamflow gauge at all?"

The second is the holy grail of operational hydrology, because most rivers in
India are ungauged - you cannot install a gauge on every stream. A model that
passes the PUB test can be deployed nationwide using only freely-available
meteorology and static catchment attributes. Reporting Leave-One-Catchment-Out
(LOCO) results demonstrates real generalisation, not memorisation, and directly
supports the proposal's goal of nation-wide risk maps under PM Gati Shakti.

This is exactly the experiment Kratzert et al. (2019) used to show LSTMs predict
ungauged basins better than calibrated conceptual models.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models.metrics import compute_all_metrics

logger = logging.getLogger("streamflow_automl.evaluation.spatial_cv")


class SpatialCrossValidator:
    """Leave-One-Catchment-Out (and grouped k-fold) spatial validation."""

    def __init__(self, target_column: str = "streamflow_mm_day"):
        self.target_column = target_column
        self.results = None

    def leave_one_catchment_out(self, model_factory, data: pd.DataFrame,
                                feature_columns: list, max_catchments: int = None,
                                scaler_factory=None) -> pd.DataFrame:
        """
        For each catchment: train on ALL OTHER catchments, test on the held-out one.

        model_factory : callable returning a fresh, unfitted model each call
                        (so no information leaks between folds).
        scaler_factory: optional callable returning a fresh scaler; fitted on the
                        training catchments only (prevents test-set leakage).
        """
        catchments = data["catchment_id"].unique()
        if max_catchments:
            catchments = catchments[:max_catchments]

        rows = []
        for held_out in catchments:
            train_df = data[data["catchment_id"] != held_out]
            test_df = data[data["catchment_id"] == held_out]
            if len(test_df) < 30 or len(train_df) < 100:
                continue

            X_train = train_df[feature_columns].values
            y_train = train_df[self.target_column].values
            X_test = test_df[feature_columns].values
            y_test = test_df[self.target_column].values

            if scaler_factory is not None:
                scaler = scaler_factory()
                X_train = scaler.fit_transform(X_train)  # fit on TRAIN basins only
                X_test = scaler.transform(X_test)

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = np.maximum(model.predict(X_test), 0)

            metrics = compute_all_metrics(y_test, y_pred)
            metrics["held_out_catchment"] = held_out
            metrics["n_test"] = len(y_test)
            if "basin" in test_df.columns:
                metrics["basin"] = test_df["basin"].iloc[0]
            rows.append(metrics)
            logger.info(
                f"PUB fold - held out {held_out}: NSE={metrics['nse']:.3f}"
            )

        self.results = pd.DataFrame(rows)
        return self.results

    def summarize(self) -> dict:
        """
        Aggregate PUB performance.

        We report the MEDIAN NSE (robust to a few badly-predicted catchments) and
        the fraction of catchments with NSE > 0 (better than predicting the mean)
        and NSE > 0.5 (a common "acceptable model" threshold in hydrology, Moriasi
        et al. 2007). These are the standard summary statistics in CAMELS papers.
        """
        if self.results is None or self.results.empty:
            return {}
        nse = self.results["nse"]
        summary = {
            "n_catchments": int(len(self.results)),
            "median_nse": round(float(nse.median()), 4),
            "mean_nse": round(float(nse.mean()), 4),
            "frac_nse_above_0": round(float((nse > 0).mean()), 4),
            "frac_nse_above_0.5": round(float((nse > 0.5).mean()), 4),
            "median_kge": round(float(self.results["kge"].median()), 4),
        }
        if "basin" in self.results.columns:
            summary["per_basin_median_nse"] = {
                basin: round(float(grp["nse"].median()), 4)
                for basin, grp in self.results.groupby("basin")
            }
        return summary
