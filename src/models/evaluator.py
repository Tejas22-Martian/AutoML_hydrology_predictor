"""
Model evaluation and comparison for streamflow predictions.

Provides comprehensive evaluation including overall metrics, extreme event
performance, per-catchment analysis, and model comparison tables.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.advanced_metrics import compute_signatures
from src.models.metrics import compute_all_metrics, compute_extreme_metrics

logger = logging.getLogger("streamflow_automl.models.evaluator")


class ModelEvaluator:
    """Evaluate and compare streamflow prediction models."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.evaluation_results = {}

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> dict:
        """Full evaluation of a single model."""
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        overall = compute_all_metrics(y_test, y_pred)

        extreme = compute_extreme_metrics(
            y_test, y_pred,
            flood_percentile=self.config.get("extreme_event_thresholds", {})
                .get("flood", {}).get("percentile", 95),
            drought_percentile=self.config.get("extreme_event_thresholds", {})
                .get("drought", {}).get("percentile", 5),
        )

        flow_regime = self._evaluate_flow_regimes(y_test, y_pred)
        signatures = compute_signatures(y_test, y_pred)

        result = {
            "model_name": model_name,
            "overall_metrics": overall,
            "extreme_event_metrics": extreme,
            "flow_regime_metrics": flow_regime,
            "hydrological_signatures": signatures,
            "n_test_samples": len(y_test),
            "predictions": {"y_test": y_test, "y_pred": y_pred},
        }

        self.evaluation_results[model_name] = result

        logger.info(
            f"{model_name} evaluation: "
            f"NSE={overall['nse']:.4f}, KGE={overall['kge']:.4f}, "
            f"RMSE={overall['rmse']:.4f}"
        )
        return result

    def evaluate_per_catchment(
        self,
        model,
        test_data: pd.DataFrame,
        feature_columns: list,
        target_column: str = "streamflow_mm_day",
    ) -> pd.DataFrame:
        """Evaluate model performance for each catchment separately."""
        results = []
        for cid, group in test_data.groupby("catchment_id"):
            if len(group) < 10:
                continue

            X = group[feature_columns].values
            y = group[target_column].values
            y_pred = np.maximum(model.predict(X), 0)

            metrics = compute_all_metrics(y, y_pred)
            metrics["catchment_id"] = cid
            metrics["n_samples"] = len(y)
            results.append(metrics)

        return pd.DataFrame(results)

    def compare_models(self, results: dict) -> pd.DataFrame:
        """Create a comparison table across multiple models."""
        rows = []
        for name, result in results.items():
            row = {"model": name}
            row.update(result.get("overall_metrics", {}))
            if "training_time_s" in result:
                row["training_time_s"] = result["training_time_s"]
            rows.append(row)

        comparison = pd.DataFrame(rows).set_index("model")
        comparison = comparison.sort_values("nse", ascending=False)

        logger.info(f"\nModel Comparison:\n{comparison.to_string()}")
        return comparison

    def _evaluate_flow_regimes(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """Evaluate performance across different flow regimes."""
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        thresholds = np.percentile(y_true, percentiles)
        regime_names = [
            "very_low", "low", "below_median",
            "above_median", "high", "very_high",
        ]

        results = {}
        for i, name in enumerate(regime_names):
            mask = (y_true >= thresholds[i]) & (y_true < thresholds[i + 1])
            if np.sum(mask) < 5:
                continue
            results[name] = compute_all_metrics(y_true[mask], y_pred[mask])
            results[name]["n_samples"] = int(np.sum(mask))

        return results

    def generate_report(self, output_path: str) -> None:
        """Generate a JSON evaluation report."""
        report = {}
        for model_name, result in self.evaluation_results.items():
            model_report = {
                k: v for k, v in result.items() if k != "predictions"
            }
            report[model_name] = model_report

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {output_path}")
