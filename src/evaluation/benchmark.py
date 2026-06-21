"""
Benchmark the AutoML model against the bundled regional-LSTM predictions.

CAMELS-IND (FOSEE) ships ``lstm_pred_streamflow.csv`` — a regionally trained LSTM
rainfall-runoff model's predictions. Comparing our AutoML model against it on the
held-out test period, per catchment, is a strong, literature-aligned baseline
(Kratzert et al. showed LSTMs match/beat calibrated conceptual models on CAMELS).
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.models.metrics import (
    kling_gupta_efficiency,
    nash_sutcliffe_efficiency,
)

logger = logging.getLogger("streamflow_automl.evaluation.benchmark")


class LSTMBenchmark:
    """Per-catchment NSE/KGE comparison: AutoML vs regional-LSTM."""

    def __init__(self, test_start: str, test_end: str, min_obs: int = 30):
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)
        self.min_obs = min_obs

    def compare(self, automl_preds: pd.DataFrame, lstm_long: pd.DataFrame) -> dict:
        """Compare predictions against shared observations, catchment by catchment.

        Args:
            automl_preds: columns [date, catchment_id, y_true, y_pred].
            lstm_long:    columns [date, catchment_id, lstm_mm_day].
        """
        preds = automl_preds.copy()
        preds["date"] = pd.to_datetime(preds["date"])
        lstm = lstm_long.copy()
        lstm["date"] = pd.to_datetime(lstm["date"])
        lstm = lstm[(lstm["date"] >= self.test_start) & (lstm["date"] <= self.test_end)]

        merged = preds.merge(lstm, on=["date", "catchment_id"], how="inner")
        merged = merged.dropna(subset=["y_true", "y_pred", "lstm_mm_day"])

        rows = []
        for cid, g in merged.groupby("catchment_id"):
            if len(g) < self.min_obs:
                continue
            yt = g["y_true"].to_numpy()
            rows.append({
                "catchment_id": cid,
                "n_obs": int(len(g)),
                "automl_nse": nash_sutcliffe_efficiency(yt, g["y_pred"].to_numpy()),
                "automl_kge": kling_gupta_efficiency(yt, g["y_pred"].to_numpy()),
                "lstm_nse": nash_sutcliffe_efficiency(yt, g["lstm_mm_day"].to_numpy()),
                "lstm_kge": kling_gupta_efficiency(yt, g["lstm_mm_day"].to_numpy()),
            })

        per_catchment = pd.DataFrame(rows)
        if per_catchment.empty:
            logger.warning("Benchmark produced no comparable catchments")
            return {"n_catchments": 0, "automl": {}, "lstm": {}, "per_catchment": []}

        wins = int((per_catchment["automl_nse"] > per_catchment["lstm_nse"]).sum())
        summary = {
            "n_catchments": int(len(per_catchment)),
            "automl_win_rate_nse": round(wins / len(per_catchment), 3),
            "automl": self._agg(per_catchment, "automl"),
            "lstm": self._agg(per_catchment, "lstm"),
            "per_catchment": per_catchment.round(4).to_dict(orient="records"),
        }
        return summary

    @staticmethod
    def _agg(df: pd.DataFrame, prefix: str) -> dict:
        return {
            "median_nse": round(float(df[f"{prefix}_nse"].median()), 4),
            "mean_nse": round(float(df[f"{prefix}_nse"].mean()), 4),
            "median_kge": round(float(df[f"{prefix}_kge"].median()), 4),
            "mean_kge": round(float(df[f"{prefix}_kge"].mean()), 4),
        }

    @staticmethod
    def save_report(summary: dict, path: str) -> None:
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Benchmark report saved to {path}")
