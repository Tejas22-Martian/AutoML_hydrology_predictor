# AutoML-Based Streamflow and Extreme Event Prediction Framework for Indian Catchments

An automated, reproducible, and India-specific predictive framework to forecast daily
streamflow and extreme hydrological events (floods and droughts) across 242 catchments
in Peninsular India using the CAMELS-IND dataset.

## Project Overview

Traditional hydrological models require expert knowledge and manual calibration. This
framework leverages AutoML to automate preprocessing, model selection, and hyperparameter
tuning — producing catchment-level risk maps that support flood preparedness, drought
mitigation, and infrastructure planning under PM Gati Shakti.

## Key Features

- **Automated ML Pipeline**: End-to-end pipeline from raw CAMELS-IND data to risk maps
- **Multiple Model Support**: Random Forest, XGBoost, LightGBM, Neural Networks (MLP),
  and an **LSTM** deep-learning rainfall-runoff model
- **Genuine Bayesian Optimization**: Optuna TPE sampler (not random search) with
  optimisation-history and hyperparameter-importance figures
- **Imbalanced Data Handling**: SMOGN for handling extreme event imbalance in regression
- **Hydrological Signatures**: log-NSE, %FHV (flood bias), %FLV (drought bias),
  %FMS (flashiness), baseflow index, peak-timing — not just NSE/KGE
- **Uncertainty Quantification**: Conformalized Quantile Regression for calibrated
  90% prediction intervals (PICP / MPIW)
- **Spatial Validation (PUB)**: Leave-One-Catchment-Out for prediction in ungauged basins
- **Interpretability**: SHAP values and Partial Dependence Plots for feature importance
- **Risk Mapping**: Catchment-scale flood/drought risk classification and visualization

> **New to the project / preparing for a viva?** Read
> [`IMPROVEMENTS.md`](IMPROVEMENTS.md) — it explains every advanced technique, the
> theory behind it, and how to defend it.

## Study Area

242 catchments across Peninsular India from the CAMELS-IND dataset, focusing on:
- Krishna Basin
- Godavari Basin
- Cauvery Basin
- Mahanadi Basin

## Project Structure

```
├── configs/                  # Configuration files
│   ├── config.yaml          # Main configuration
│   └── model_configs.yaml   # Model hyperparameter spaces
├── data/
│   ├── raw/                 # Raw CAMELS-IND data
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # Shapefiles, basin boundaries
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_interpretability.ipynb
│   └── 05_risk_mapping.ipynb
├── src/
│   ├── data/                # Data loading and preprocessing
│   ├── features/            # Feature engineering
│   ├── models/              # Model training and AutoML
│   ├── visualization/       # Plotting and risk maps
│   └── utils/               # Utility functions
├── outputs/
│   ├── models/              # Saved trained models
│   ├── figures/             # Generated plots
│   ├── risk_maps/           # Flood/drought risk maps
│   └── reports/             # Evaluation reports
├── tests/                   # Unit tests
├── main.py                  # Main pipeline entry point
├── requirements.txt         # Python dependencies
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/automl-hydro.git
cd automl-hydro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Full Pipeline

The pipeline runs **entirely on the real CAMELS-IND (FOSEE) dataset** — there is
no synthetic data. It expects the dataset next to the repo at
`../FOSEE/CAMELS_IND_Catchments_Streamflow_Sufficient/` (configurable in
`configs/config.yaml`).

```bash
# Run the complete pipeline (data → train → evaluate → benchmark → uncertainty
# → interpret → risk_maps)
python main.py --config configs/config.yaml

# Run specific phases
python main.py --config configs/config.yaml --phase data
python main.py --config configs/config.yaml --phase train
python main.py --config configs/config.yaml --phase evaluate
python main.py --config configs/config.yaml --phase benchmark   # vs regional-LSTM
python main.py --config configs/config.yaml --phase risk_maps
python main.py --config configs/config.yaml --phase spatial_cv  # PUB / ungauged
```

The pipeline runs on **CPU or GPU automatically** — scikit-learn models use CPU,
and the optional LSTM uses CUDA when a GPU is available, else CPU. For full
training, inference, and deployment instructions see
[DEPLOYMENT.md](DEPLOYMENT.md).

### Inference with the fine-tuned model (no training needed)

The repo ships a ready-to-use bundle in `outputs/models/`:
`best_model.joblib` (the fine-tuned model) + `preprocessor.joblib` (the fitted
feature scaler/columns). `predict.py` runs the full path — load forcings →
engineer features → apply the fitted preprocessor → predict — and writes a CSV.

```bash
# Predict daily streamflow (mm/day) for one catchment, using the shipped model
python predict.py --catchment 03001 --output predictions.csv
```

Output columns: `date, catchment_id, predicted_streamflow_mm_day`.

**Run on your own CAMELS-IND copy / a different release** — point a config's
`data:` paths at it and pass `--config`:

```bash
python predict.py --catchment 03001 --config configs/my_data.yaml --output predictions.csv
```

Inference is **CPU-only** for the shipped MLP model — no GPU required. See
[DEPLOYMENT.md](DEPLOYMENT.md) for serving the model behind an API.

### Training / fine-tuning your own model

Re-run the AutoML search to fit a fresh model on your data or scope. This
regenerates a matched `best_model.joblib` + `preprocessor.joblib` pair.

```bash
# 1. Point configs/config.yaml `data:` at your CAMELS-IND copy, then:
python main.py --config configs/config.yaml            # full pipeline
python main.py --config configs/config.yaml --phase train   # train only
```

To **fine-tune the search** (edit `configs/`, no code changes):

- `configs/config.yaml`
  - `data.focus_basins` — which basins the pooled model trains on
  - `data.train_start/train_end/test_start/test_end` — temporal split
  - `training.use_deliverable_profile: false` — run the **exhaustive** Optuna
    search (more trials/rows; slower but stronger). The `deliverable:` block caps
    trials/rows for a tractable run.
  - `features.*` — lag days, rolling windows, attribute set
- `configs/model_configs.yaml` — per-model hyperparameter search spaces; set
  `enabled: true/false` to include/exclude a model (e.g. enable the GPU `lstm`).

GPU is used automatically for the LSTM when a CUDA build of PyTorch is installed;
everything else trains on CPU. See [DEPLOYMENT.md](DEPLOYMENT.md).

### Individual Components

```python
import yaml
from src.data.loader import CAMELSIndDataLoader
from src.data.preprocessor import StreamflowPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.automl_trainer import AutoMLTrainer

cfg = yaml.safe_load(open("configs/config.yaml"))

# Load real CAMELS-IND data (forcings + observed streamflow in mm/day + attrs)
loader = CAMELSIndDataLoader(cfg["data"],
                             attribute_features=cfg["features"]["catchment_attributes"])
focus = loader.get_basin_catchments(cfg["data"]["focus_basins"])
data = loader.load_all_catchments(focus)

# Feature-engineer (per-catchment) and preprocess (temporal split, no leakage)
data = FeatureEngineer(cfg["features"]).transform(data)
pre = StreamflowPreprocessor(cfg["preprocessing"])
X_train, X_test, y_train, y_test = pre.fit_transform(
    data,
    train_start=cfg["data"]["train_start"], train_end=cfg["data"]["train_end"],
    test_start=cfg["data"]["test_start"], test_end=cfg["data"]["test_end"],
)

# AutoML (Optuna TPE over RF/XGBoost/LightGBM/MLP/LSTM)
trainer = AutoMLTrainer(config=yaml.safe_load(open("configs/model_configs.yaml")))
best_model = trainer.fit(X_train, y_train, n_trials=15, cv_folds=3)
```

## Methodology

1. **Data Acquisition**: Load CAMELS-IND dataset (meteorological forcings, streamflow,
   catchment attributes)
2. **Preprocessing**: Handle missing values, normalize features, apply SMOGN for
   extreme event balancing
3. **Feature Engineering**: Lag features, rolling statistics, seasonal indicators,
   catchment attributes integration
4. **AutoML Training**: Bayesian Optimization over Random Forest, XGBoost, LightGBM,
   Neural Networks (MLP), and an LSTM
5. **Evaluation**: NSE, KGE, RMSE, PBIAS metrics with k-fold cross-validation
6. **Interpretability**: SHAP values, Partial Dependence Plots, feature importance
7. **Risk Mapping**: Classify catchments into flood/drought risk levels, generate maps

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| NSE    | Nash-Sutcliffe Efficiency (-∞ to 1) |
| KGE    | Kling-Gupta Efficiency (-∞ to 1) |
| RMSE   | Root Mean Square Error |
| PBIAS  | Percent Bias |
| R²     | Coefficient of Determination |

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| M1–M2 | Months 1-2 | Data acquisition, preprocessing |
| M3–M4 | Months 3-4 | AutoML model building & hyperparameter optimization |
| M5    | Month 5 | Model validation, SHAP-based interpretability |
| M6    | Month 6 | Final risk maps, documentation, open-source release |

## Impact & SDG Alignment

- **SDG 6** (Clean Water): Improved water resource management
- **SDG 11** (Sustainable Cities): Urban flood risk mitigation
- **SDG 13** (Climate Action): Climate-adaptive infrastructure planning
- **SDG 15** (Life on Land): Drought monitoring for ecosystem protection

## Scope & Limitations

Read this before relying on the predictions.

- **Autoregressive, one-day-ahead.** The model uses lagged/rolling streamflow as
  features, so it predicts *next-day* flow given recent flow + meteorology. Much
  of its skill is near-term persistence plus corrections — it is **not** a
  long-horizon forecast or a fully physics-based rainfall-runoff emulator.
- **Trained on four focus basins.** The pooled model is fit/tuned on Krishna,
  Godavari, Cauvery, and Mahanadi (~131 catchments). Expect strong skill on
  in-distribution catchments (e.g. NSE ≈ 1.0) and **magnitude bias on far
  out-of-distribution catchments** — predictions stay well-correlated in shape
  but can be off in scale (the prediction-in-ungauged-basins / PUB setting).
- **Uneven performance across flow regimes.** Aggregate NSE/KGE are dominated by
  high-flow variance; the model is strong on high flows but weaker on
  below-median low flows. Low-flow / drought skill is the main area for
  improvement (e.g. log-space or flow-stratified training loss).
- **Benchmark caveat.** The bundled regional-LSTM comparison is not
  apples-to-apples (different forecasting setup); our high win rate reflects the
  autoregressive next-day setup, not a better long-range emulator.
- **Input requirements.** Inference needs the CAMELS-IND forcing variables and
  static attributes used in training; new catchments must provide the same
  schema. The deliverable profile caps Optuna trials/rows — disable it for the
  exhaustive search.

## Data & Citation

This project uses the **CAMELS-IND** dataset, which is openly available for
academic and research use and **must be cited**. The dataset is not bundled in
full here — see [`data/README.md`](data/README.md) for how to obtain it and the
required citation. Per the dataset disclaimer, all interpretations and
conclusions are the user's own.

## License

This project is released under Creative Commons Attribution 4.0 (CC BY 4.0) and
MIT License for code components. The CAMELS-IND dataset retains its own terms.
