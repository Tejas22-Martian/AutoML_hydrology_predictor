"""
Main pipeline entry point for AutoML Streamflow Prediction.

Orchestrates the full workflow: data loading → preprocessing → feature engineering
→ model training → evaluation → interpretability → risk mapping.

Usage:
    python main.py --config configs/config.yaml
    python main.py --config configs/config.yaml --phase train

Runs entirely on the real CAMELS-IND (FOSEE) dataset. No synthetic data.
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib

import numpy as np
import pandas as pd

from src.data.loader import CAMELSIndDataLoader
from src.data.preprocessor import StreamflowPreprocessor
from src.evaluation.benchmark import LSTMBenchmark
from src.evaluation.spatial_cv import SpatialCrossValidator
from src.features.engineer import FeatureEngineer
from src.models.automl_trainer import AutoMLTrainer
from src.models.evaluator import ModelEvaluator
from src.models.interpretability import SHAPExplainer
from src.models.metrics import compute_all_metrics
from src.models.uncertainty import QuantileIntervalEstimator
from src.utils.helpers import load_config, set_seed, setup_logging, ensure_dir
from src.visualization.plots import HydrologicalPlotter
from src.visualization.risk_maps import RiskMapGenerator

logger = logging.getLogger("streamflow_automl")


class StreamflowPipeline:
    """End-to-end pipeline for streamflow prediction and risk mapping."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model_configs = load_config("configs/model_configs.yaml")
        set_seed(self.config["project"]["seed"])

        self.loader = CAMELSIndDataLoader(
            self.config["data"],
            attribute_features=self.config["features"].get("catchment_attributes"),
        )
        self.feature_engineer = FeatureEngineer(self.config.get("features", {}))
        self.preprocessor = StreamflowPreprocessor(self.config.get("preprocessing", {}))
        self.trainer = AutoMLTrainer(config=self.model_configs)
        self.evaluator = ModelEvaluator(self.config.get("evaluation", {}))
        self.plotter = HydrologicalPlotter(self.config["output"]["figures_dir"])
        self.risk_mapper = RiskMapGenerator({
            **self.config.get("risk_mapping", {}),
            "output_dir": self.config["output"]["risk_maps_dir"],
        })

        for key in ["models_dir", "figures_dir", "risk_maps_dir", "reports_dir"]:
            ensure_dir(self.config["output"][key])

        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.feature_names = None
        self.train_ids = None
        self.report_ids = None

    def run_full_pipeline(self) -> dict:
        """Execute all pipeline phases."""
        logger.info("=" * 60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("=" * 60)

        self.phase_data()
        self.phase_train()
        results = self.phase_evaluate()
        self.phase_benchmark()
        self.phase_uncertainty()
        self.phase_interpret()
        self.phase_risk_maps()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        return results

    def phase_data(self) -> None:
        """Phase 1: Data loading, feature engineering, and preprocessing.

        Trains on the focus basins (catchments with sufficient observed record);
        retains the full catchment list for basin-wide risk mapping.
        """
        logger.info("-" * 40)
        logger.info("PHASE 1: DATA LOADING & PREPROCESSING")
        logger.info("-" * 40)

        data_cfg = self.config["data"]
        all_catchments = self.loader.list_catchments()
        if not all_catchments:
            raise RuntimeError(
                f"No CAMELS-IND catchments found under {data_cfg['forcings_dir']}. "
                "Check the FOSEE dataset paths in configs/config.yaml."
            )
        logger.info(f"Found {len(all_catchments)} catchments in dataset")

        focus = self.loader.get_basin_catchments(data_cfg["focus_basins"])
        self.train_ids = self.loader.filter_trainable(
            focus,
            min_samples=data_cfg.get("min_train_samples", 1095),
            start=data_cfg["train_start"],
            end=data_cfg["train_end"],
        )
        self.report_ids = (
            all_catchments if data_cfg.get("report_all_catchments", True)
            else self.train_ids
        )
        logger.info(
            f"Focus basins {data_cfg['focus_basins']}: "
            f"{len(self.train_ids)} trainable catchments; "
            f"risk maps will cover {len(self.report_ids)} catchments"
        )

        raw = self.loader.load_all_catchments(self.train_ids)
        self.data = self.feature_engineer.transform(raw)
        self.feature_names = self.feature_engineer.get_feature_names(self.data)
        logger.info(f"Engineered {len(self.feature_names)} features")

        preprocess_cfg = self.config.get("preprocessing", {})
        preprocess_cfg["target_column"] = data_cfg.get(
            "target_column", "streamflow_mm_day"
        )
        # Bound preprocessing/SMOGN memory: cap train rows here (before scaling/
        # SMOGN) when the deliverable profile is active, not just at fit time.
        training_cfg = self.config.get("training", {})
        if training_cfg.get("use_deliverable_profile", False):
            preprocess_cfg["max_train_rows"] = training_cfg.get("deliverable", {}).get(
                "max_train_rows"
            )
        self.preprocessor = StreamflowPreprocessor(preprocess_cfg)
        self.X_train, self.X_test, self.y_train, self.y_test = (
            self.preprocessor.fit_transform(
                self.data,
                train_start=data_cfg.get("train_start"),
                train_end=data_cfg.get("train_end"),
                test_start=data_cfg.get("test_start"),
                test_end=data_cfg.get("test_end"),
            )
        )

        # Persist the fitted preprocessor so deployment (predict.py) can transform
        # raw forcings exactly as in training (scaler, feature columns, fill medians).
        models_dir = Path(self.config["output"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, models_dir / "preprocessor.joblib")
        logger.info(f"Preprocessor saved to {models_dir / 'preprocessor.joblib'}")

    def phase_train(self) -> None:
        """Phase 2: AutoML model training with Bayesian Optimization."""
        logger.info("-" * 40)
        logger.info("PHASE 2: AUTOML MODEL TRAINING")
        logger.info("-" * 40)

        training_cfg = self.config.get("training", {})
        n_trials = training_cfg.get("n_optimization_trials", 50)
        cv_folds = training_cfg.get("cv_folds", 5)

        X_train, y_train = self.X_train, self.y_train
        if training_cfg.get("use_deliverable_profile", False):
            prof = training_cfg.get("deliverable", {})
            n_trials = prof.get("n_optimization_trials", n_trials)
            cv_folds = prof.get("cv_folds", cv_folds)
            X_train, y_train = self._subsample_temporal(
                X_train, y_train, prof.get("max_train_rows")
            )
            logger.info(
                f"Deliverable profile: n_trials={n_trials}, cv_folds={cv_folds}, "
                f"train_rows={len(y_train)}"
            )

        self.best_model = self.trainer.fit(
            X_train, y_train, n_trials=n_trials, cv_folds=cv_folds,
        )

        model_path = Path(self.config["output"]["models_dir"]) / "best_model.joblib"
        self.trainer.save_model(str(model_path))

        # Visualise the Bayesian Optimization search for each tuned model
        for name, study in self.trainer.studies.items():
            try:
                self.plotter.plot_optimization_history(study, model_name=name)
            except Exception as e:
                logger.warning(f"Could not plot optimisation history for {name}: {e}")

    @staticmethod
    def _subsample_temporal(X: np.ndarray, y: np.ndarray, max_rows) -> tuple:
        """Evenly stride-subsample rows to a cap, preserving array order."""
        if not max_rows or len(y) <= max_rows:
            return X, y
        idx = np.linspace(0, len(y) - 1, int(max_rows)).astype(int)
        return X[idx], y[idx]

    def phase_uncertainty(self) -> None:
        """Phase 3b: Quantile-regression prediction intervals (uncertainty)."""
        logger.info("-" * 40)
        logger.info("PHASE 3b: UNCERTAINTY QUANTIFICATION")
        logger.info("-" * 40)

        eval_cfg = self.config.get("evaluation", {})
        lower = eval_cfg.get("interval_lower", 0.05)
        upper = eval_cfg.get("interval_upper", 0.95)

        estimator = QuantileIntervalEstimator(lower=lower, upper=upper)
        estimator.fit(self.X_train, self.y_train)
        lo, med, hi = estimator.predict_interval(self.X_test)
        interval_metrics = estimator.evaluate_interval(self.X_test, self.y_test)

        logger.info(
            f"Prediction interval: nominal={interval_metrics['nominal_coverage']:.0%}, "
            f"observed PICP={interval_metrics['picp_observed_coverage']:.1%}, "
            f"mean width={interval_metrics['mpiw_mean_interval_width']:.3f}"
        )

        self.plotter.plot_prediction_interval(
            None, self.y_test, lo, med, hi,
            picp=interval_metrics["picp_observed_coverage"],
        )

        report_path = (
            Path(self.config["output"]["reports_dir"]) / "uncertainty_report.json"
        )
        import json
        with open(report_path, "w") as f:
            json.dump(interval_metrics, f, indent=2)
        logger.info(f"Uncertainty report saved to {report_path}")

    def phase_spatial_cv(self) -> None:
        """Optional: Leave-One-Catchment-Out (Prediction in Ungauged Basins)."""
        logger.info("-" * 40)
        logger.info("PHASE: SPATIAL CROSS-VALIDATION (PUB)")
        logger.info("-" * 40)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler

        feature_cols = self.feature_engineer.get_feature_names(self.data)
        validator = SpatialCrossValidator(
            target_column=self.config["data"].get("target_column", "streamflow_mm_day")
        )
        validator.leave_one_catchment_out(
            model_factory=lambda: RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=42
            ),
            data=self.data,
            feature_columns=feature_cols,
            scaler_factory=StandardScaler,
        )
        summary = validator.summarize()
        logger.info(f"PUB summary: {summary}")

        report_path = (
            Path(self.config["output"]["reports_dir"]) / "spatial_cv_report.json"
        )
        import json
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Spatial CV report saved to {report_path}")

    def phase_evaluate(self) -> dict:
        """Phase 3: Model evaluation and visualization."""
        logger.info("-" * 40)
        logger.info("PHASE 3: MODEL EVALUATION")
        logger.info("-" * 40)

        result = self.evaluator.evaluate(
            self.best_model, self.X_test, self.y_test,
            model_name=self.trainer.best_model_name,
        )

        y_pred = result["predictions"]["y_pred"]

        # Align dates to the test rows via the preprocessor's retained metadata.
        test_meta = self.preprocessor.test_meta
        has_dates = test_meta is not None and "date" in test_meta.columns
        dates = test_meta["date"].values if has_dates else np.arange(len(self.y_test))

        self.plotter.plot_hydrograph(dates, self.y_test, y_pred)
        self.plotter.plot_scatter(self.y_test, y_pred)
        self.plotter.plot_flow_duration_curve(self.y_test, y_pred)

        if has_dates:
            self.plotter.plot_monthly_performance(dates, self.y_test, y_pred)

        report_path = (
            Path(self.config["output"]["reports_dir"]) / "evaluation_report.json"
        )
        self.evaluator.generate_report(str(report_path))

        logger.info(f"Best model: {self.trainer.best_model_name}")
        logger.info(f"Test NSE: {result['overall_metrics']['nse']:.4f}")
        logger.info(f"Test KGE: {result['overall_metrics']['kge']:.4f}")

        return result

    def phase_interpret(self) -> None:
        """Phase 4: SHAP-based interpretability analysis."""
        logger.info("-" * 40)
        logger.info("PHASE 4: INTERPRETABILITY (SHAP)")
        logger.info("-" * 40)

        feature_names = self.preprocessor.feature_columns or [
            f"feature_{i}" for i in range(self.X_test.shape[1])
        ]

        explainer = SHAPExplainer(
            self.best_model,
            feature_names,
            output_dir=self.config["output"]["figures_dir"],
        )

        explainer.compute_shap_values(self.X_test, sample_size=200)
        explainer.plot_feature_importance(top_n=20)

        try:
            explainer.plot_summary(self.X_test)
        except Exception as e:
            logger.warning(f"SHAP summary plot failed: {e}")

        top_features = explainer.plot_feature_importance(top_n=5)
        for feat in top_features.head(3)["feature"]:
            try:
                explainer.plot_dependence(feat, self.X_test)
            except Exception:
                pass

    def phase_benchmark(self) -> dict:
        """Phase 3c: Benchmark our AutoML model against the regional-LSTM baseline."""
        logger.info("-" * 40)
        logger.info("PHASE 3c: REGIONAL-LSTM BENCHMARK")
        logger.info("-" * 40)

        if not self.config.get("evaluation", {}).get("benchmark_lstm", True):
            logger.info("LSTM benchmark disabled in config; skipping")
            return {}

        data_cfg = self.config["data"]
        lstm_long = self.loader.load_lstm_predictions(self.train_ids)
        if lstm_long.empty:
            logger.warning("No LSTM baseline available; skipping benchmark")
            return {}

        # Our model's predictions on the test set, with date + catchment metadata.
        y_pred = self.best_model.predict(self.X_test)
        preds = self.preprocessor.test_meta.copy()
        preds["y_true"] = self.y_test
        preds["y_pred"] = np.maximum(y_pred, 0)

        benchmark = LSTMBenchmark(
            test_start=data_cfg["test_start"], test_end=data_cfg["test_end"]
        )
        summary = benchmark.compare(preds, lstm_long)
        report_path = (
            Path(self.config["output"]["reports_dir"]) / "benchmark_report.json"
        )
        benchmark.save_report(summary, str(report_path))
        try:
            self.plotter.plot_benchmark(summary, output_name="benchmark_lstm_vs_automl")
        except Exception as e:
            logger.warning(f"Benchmark plot failed: {e}")

        logger.info(
            f"Benchmark (median NSE): AutoML={summary['automl']['median_nse']:.3f} "
            f"vs regional-LSTM={summary['lstm']['median_nse']:.3f} "
            f"across {summary['n_catchments']} catchments"
        )
        return summary

    def phase_risk_maps(self) -> None:
        """Phase 5: Generate catchment-level flood/drought risk maps (all catchments)."""
        logger.info("-" * 40)
        logger.info("PHASE 5: RISK MAP GENERATION")
        logger.info("-" * 40)

        data_cfg = self.config["data"]
        catchment_attrs = self.loader.get_attributes()
        test_start = pd.Timestamp(data_cfg["test_start"])

        # Stream per-catchment: engineer + scale + predict one catchment at a time
        # so memory stays bounded (engineering all 242 at once is ~3.6M rows -> OOM).
        # Only the 1-D predictions and ids are retained.
        pred_chunks, id_chunks = [], []
        for cid in self.report_ids:
            try:
                raw = self.loader.load_catchment(cid)
                eng = self.feature_engineer.transform(raw)
                eng = eng[eng["date"] >= test_start]
                if eng.empty:
                    continue
                X = self.preprocessor.transform(eng)
                pred_chunks.append(self.best_model.predict(X))
                id_chunks.append(eng["catchment_id"].values)
            except Exception as e:
                logger.warning(f"Risk mapping skipped catchment {cid}: {e}")

        if not pred_chunks:
            logger.warning("No test-period rows for risk mapping; skipping")
            return

        predictions = np.concatenate(pred_chunks)
        catchment_ids = np.concatenate(id_chunks)

        risk_df = self.risk_mapper.compute_risk_scores(
            self.best_model, None, catchment_ids, catchment_attrs,
            predictions=predictions,
        )

        if not risk_df.empty:
            self.risk_mapper.generate_static_map(risk_df, risk_type="flood")
            self.risk_mapper.generate_static_map(risk_df, risk_type="drought")
            self.risk_mapper.generate_interactive_map(risk_df)
            self.risk_mapper.export_geojson(risk_df)
            self.risk_mapper.generate_summary_report(risk_df)
            self.risk_mapper.generate_choropleth(
                risk_df, shapefile=data_cfg.get("shapefile"), risk_type="flood"
            )


def main():
    parser = argparse.ArgumentParser(
        description="AutoML Streamflow Prediction Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "data", "train", "evaluate", "benchmark", "uncertainty",
                 "interpret", "risk_maps", "spatial_cv"],
        help="Pipeline phase to run",
    )

    args = parser.parse_args()
    setup_logging()

    pipeline = StreamflowPipeline(args.config)

    if args.phase == "all":
        pipeline.run_full_pipeline()
    else:
        phase_method = getattr(pipeline, f"phase_{args.phase}")
        needs_train = (
            "evaluate", "benchmark", "uncertainty", "interpret", "risk_maps",
        )
        if args.phase != "data":
            pipeline.phase_data()
        if args.phase in needs_train:
            pipeline.phase_train()
        phase_method()


if __name__ == "__main__":
    main()
