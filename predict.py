"""Inference / deployment entry point.

Loads the fine-tuned model + fitted preprocessor produced by the training
pipeline and predicts daily streamflow (mm/day) for a catchment, writing a CSV.

Device is selected automatically: scikit-learn models run on CPU; the optional
LSTM uses CUDA when a GPU is available and CPU otherwise (see
src/models/lstm_model.py::_resolve_device). No GPU is required for the shipped
MLP model.

Usage:
    python predict.py --catchment 03001 --output predictions.csv
    python predict.py --catchment 03001 --config configs/config.yaml
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.data.loader import CAMELSIndDataLoader
from src.features.engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("predict")


def load_bundle(models_dir: Path):
    """Load the trained model and fitted preprocessor."""
    model_blob = joblib.load(models_dir / "best_model.joblib")
    model = model_blob["model"] if isinstance(model_blob, dict) else model_blob
    pre_path = models_dir / "preprocessor.joblib"
    if not pre_path.exists():
        raise FileNotFoundError(
            f"{pre_path} not found. Run the training pipeline once "
            "(`python main.py --config configs/config.yaml`) to produce the "
            "matched model + preprocessor bundle."
        )
    preprocessor = joblib.load(pre_path)
    return model, preprocessor


def predict_catchment(catchment_id: str, config: dict) -> pd.DataFrame:
    """Predict streamflow for one catchment using the configured CAMELS-IND data."""
    loader = CAMELSIndDataLoader(
        config["data"],
        attribute_features=config["features"].get("catchment_attributes"),
    )
    engineer = FeatureEngineer(config["features"])
    model, preprocessor = load_bundle(Path(config["output"]["models_dir"]))

    raw = loader.load_catchment(catchment_id)
    engineered = engineer.transform(raw)
    if engineered.empty:
        raise ValueError(f"No usable rows for catchment {catchment_id}")

    X = preprocessor.transform(engineered)
    preds = model.predict(X).clip(min=0)

    return pd.DataFrame({
        "date": engineered["date"].values,
        "catchment_id": catchment_id,
        "predicted_streamflow_mm_day": preds.round(4),
    })


def main():
    parser = argparse.ArgumentParser(description="Streamflow inference")
    parser.add_argument("--catchment", required=True, help="CAMELS-IND gauge id")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    result = predict_catchment(args.catchment, config)
    result.to_csv(args.output, index=False)
    logger.info(
        f"Wrote {len(result)} predictions for catchment {args.catchment} -> {args.output}"
    )


if __name__ == "__main__":
    main()
