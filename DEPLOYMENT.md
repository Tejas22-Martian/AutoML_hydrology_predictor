# Deployment & Usage Guide

How to run the fine-tuning pipeline and use the fine-tuned model for inference.
The code runs on **CPU or GPU automatically** — no manual switch needed.

## 1. Install

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the core (CPU) stack. PyTorch is **optional**: it is only
needed to train the deep-learning LSTM. If PyTorch is absent the pipeline simply
skips the LSTM and uses the tree/MLP models (graceful degradation). The shipped
fine-tuned model is a scikit-learn **MLP**, so **inference needs only CPU**.

### CPU vs GPU (automatic)

- **scikit-learn models** (RandomForest, XGBoost, LightGBM, MLP) run on CPU.
- **The LSTM** auto-selects the device: CUDA when a GPU is visible, otherwise CPU
  (`src/models/lstm_model.py::_resolve_device`). To train the LSTM on GPU, just
  install a CUDA build of PyTorch; nothing else changes.

```bash
# GPU training (optional): install a CUDA build of torch, then run normally
pip install torch --index-url https://download.pytorch.org/whl/cu121
python main.py --config configs/config.yaml
```

## 2. Data layout

The pipeline reads the real CAMELS-IND (FOSEE) release. Point `configs/config.yaml`
(`data:` section) at your copy:

```
<fosee_root>/
├── catchment_mean_forcings/{gauge_id}.csv
├── streamflow_timeseries/streamflow_observed.csv
├── attributes_csv/camels_ind_*.csv
└── shapefiles_catchment/catchments.shp
```

## 3. Fine-tuning pipeline (training)

```bash
python main.py --config configs/config.yaml
```

This runs data → train → evaluate → benchmark → uncertainty → interpret →
risk_maps and writes two deployable artifacts to `outputs/models/`:

- `best_model.joblib` — the selected fine-tuned model
- `preprocessor.joblib` — the fitted feature scaler + column/median state

> The `training.use_deliverable_profile` flag in `config.yaml` caps Optuna
> trials and training rows so a full real-data run is tractable. Set it to
> `false` for the exhaustive (slower) search.

Single phase (downstream phases auto-run their prerequisites):

```bash
python main.py --config configs/config.yaml --phase train
```

## 4. Inference (real deployment)

Once `best_model.joblib` **and** `preprocessor.joblib` exist, predict streamflow
for a catchment:

```bash
python predict.py --catchment 03001 --output predictions.csv
```

Output CSV columns: `date, catchment_id, predicted_streamflow_mm_day`.

`predict.py` performs the full path — load forcings → engineer features →
apply the fitted preprocessor → model predict — so raw inputs are transformed
exactly as in training.

### Note on the committed model

`outputs/models/best_model.joblib` is the **reference fine-tuned result** from our
run. For raw-input inference it must be paired with the **matching**
`preprocessor.joblib`. If you only have the model, run the pipeline once (step 3)
to regenerate a matched model + preprocessor pair, then use `predict.py`.

### Serving as an API

The model is a standard scikit-learn estimator. To serve it, wrap
`predict.predict_catchment` (or `model.predict` on a pre-built feature matrix) in
your web framework of choice. Per the project convention for ML endpoints, return
a job id and run inference asynchronously rather than blocking the request.

## 5. Reproducibility

- All splits and cross-validation are **temporal** (no shuffled leakage).
- Random seeds are set in `configs/config.yaml` (`project.seed`).
- The model search space lives in `configs/model_configs.yaml`.
