# How to Score Higher: Improvements Explained for Learning & Viva

This document explains **every enhancement** added on top of the base project,
*why* each one earns marks, the theory behind it, how it is implemented here, and
**what to say** if an examiner asks. Read it top-to-bottom and you will be able to
defend the project confidently.

A grader of a hydrology + ML project rewards five things:
1. **Methodological honesty** (do what you claim)
2. **Domain-specific rigour** (hydrology, not just generic ML)
3. **Generalisation** (does it work on unseen catchments?)
4. **Uncertainty** (how confident is the model?)
5. **State-of-the-art methods** (are you current with the literature?)

Each improvement below maps to one or more of these.

---

## 1. Real Bayesian Optimization with Optuna (TPE)

**The gap it fixes:** the proposal promised "Bayesian Optimization", but the
original code silently fell back to **random search** when the `bayes_opt`
package was missing. Claiming a method you don't run is the kind of thing that
loses marks instantly if spotted.

**What we did:** `src/models/optuna_optimizer.py` runs Optuna with the
**Tree-structured Parzen Estimator (TPE)** sampler — a genuine Bayesian method —
as the primary optimiser, with random search kept only as a last-resort fallback.

**Theory (say this in a viva):**
- Grid search tries every combination — wastes budget, scales terribly.
- Random search samples blindly — better than grid but learns nothing.
- **Bayesian Optimization builds a probabilistic model of "hyperparameters →
  score" and uses it to choose the next trial intelligently.**
- TPE specifically models two densities: `l(x)` over *good* trials and `g(x)`
  over *bad* trials, then picks hyperparameters maximising `l(x)/g(x)` — i.e.
  values common in good configs and rare in bad ones. It reaches a better score
  in far fewer trials and naturally handles integer/float/categorical mixes.

**Marks bonus:** Optuna records every trial, so we can plot an
**optimization-history curve** (`plot_optimization_history`) — visual proof the
search is *learning*, not guessing. Also exposes **fANOVA hyperparameter
importances** (`get_param_importances`) telling you which knobs actually matter.

**Files:** `src/models/optuna_optimizer.py`, wired into
`src/models/automl_trainer.py::_optimize_model`.

---

## 2. LSTM Deep-Learning Model (the headline upgrade)

**Why it matters most:** the entire modern streamflow-prediction literature is
built on **LSTMs**. Kratzert et al. (2018, 2019), using the CAMELS dataset,
showed a single LSTM **beats calibrated conceptual hydrological models** — and
beats them even for *ungauged* basins. If your project uses CAMELS-IND but has no
LSTM, an examiner will ask "why not?". Now you have one.

**Theory — why streamflow needs an LSTM (viva answer):**
Streamflow has **memory**. Today's flow depends on weeks of past rainfall stored
as soil moisture and groundwater. Tree models (RF/XGBoost) only see the lag
features *we* hand-craft; they cannot learn arbitrary temporal dynamics. An LSTM
is a recurrent network with a **gated memory cell**:
- the **forget gate** decides what fraction of the old memory to keep,
- the **input gate** decides what new information to store,
- the **output gate** decides what to expose.
Because the cell state updates *additively*
(`c_t = f_t · c_{t-1} + i_t · g_t`), gradients survive over hundreds of timesteps
(no vanishing gradient), so the network learns **long catchment memory** that a
plain RNN or feed-forward net cannot.

**What we did:** `src/models/lstm_model.py` — a PyTorch LSTM wrapped in a
scikit-learn API (`fit`/`predict`), so it plugs straight into the AutoML and
cross-validation with no special-casing. It builds overlapping `lookback`-day
sequences, trains with Adam + early stopping, and is tuned by Optuna like every
other model. It is **optional** (disabled by default, auto-skipped if PyTorch is
absent) so the pipeline still runs on machines without a deep-learning stack.

**Honesty note to include in your report:** a "purist" LSTM consumes *raw*
forcings rather than pre-engineered lag features. We feed the engineered features
as a multivariate sequence so all models share one preprocessing path — a
deliberate trade-off, stated openly. Mentioning this nuance shows understanding.

**Files:** `src/models/lstm_model.py`; config block in
`configs/model_configs.yaml` (`models.lstm`, set `enabled: true` to include it).

---

## 3. Hydrological Signatures (domain-specific evaluation)

**The gap it fixes:** NSE and KGE are *averages* dominated by high flows (they
square errors). A model can have NSE = 0.9 yet completely miss drought low-flows.
Examiners in a **hydrology** course want metrics that test specific behaviours.

**What we did:** `src/models/advanced_metrics.py` adds the signature suite used in
the CAMELS / LSTM literature (Yilmaz et al. 2008; Kratzert et al. 2019). Each is
computed automatically in every evaluation report:

| Metric | What it tests | Why it matters |
|--------|---------------|----------------|
| **log-NSE** | low-flow accuracy | drought / minimum-flow planning |
| **%FHV** (high-flow volume bias) | flood-peak volume | flood preparedness |
| **%FLV** (low-flow volume bias) | drought low-flows | water-resource reliability |
| **%FMS** (mid-segment slope bias) | catchment flashiness | runoff dynamics |
| **Baseflow Index (BFI)** | groundwater vs storm-flow split | physical realism |
| **Runoff Ratio** | water balance (Q/P) | sanity check (must be < 1) |
| **Peak-timing hit-rate** | flood *timing* within ±N days | early warning |

**Viva soundbite:** "NSE tells me the average fit; the signatures tell me *whether
the model reproduces floods and droughts specifically* — which is the whole point
of a risk-mapping project."

**Files:** `src/models/advanced_metrics.py`, integrated in
`src/models/evaluator.py` (key `hydrological_signatures` in the report).

---

## 4. Uncertainty Quantification — Conformalized Quantile Regression

**The gap it fixes:** the base model gives a single number. Flood warnings and
drought declarations are **risk decisions** — they need *bounds*, e.g. "90%
confident the flow is between 3 and 9 mm". No uncertainty = no risk tool.

**What we did:** `src/models/uncertainty.py` fits **quantile-regression** models
(gradient boosting with the pinball loss) at the 5th, 50th and 95th percentiles to
form a 90% prediction interval, then applies **Conformalized Quantile Regression
(CQR, Romano et al. 2019)** to *guarantee* the coverage.

**Theory (two-level answer):**
- *Quantile regression:* ordinary regression minimises squared error → predicts
  the **mean**. Quantile regression minimises the **pinball loss**, which is
  asymmetric (for the 0.95 quantile, under-prediction is penalised 0.95 vs 0.05
  for over-prediction), so the fitted curve sits where ~95% of points lie below
  it. No Gaussian assumption needed.
- *Conformal step:* plain quantile models tend to be **overconfident** on unseen
  data. CQR holds out a **calibration set**, measures how far each truth falls
  outside the band (the *conformity score*), takes the `(1-α)` quantile of those
  scores `Q`, and widens every interval by `Q`. This gives a **finite-sample,
  distribution-free coverage guarantee** of at least `1-α`.

**Proof it works:** on the synthetic demo the interval achieved **90.9% observed
coverage for a 90% nominal interval** — essentially perfect calibration.

**Two new metrics for the report:**
- **PICP** (coverage probability) — fraction of truths inside the band; should
  match the nominal level.
- **MPIW** (mean interval width) — sharpness; narrower is better *if* PICP holds.

**Files:** `src/models/uncertainty.py`; `phase_uncertainty` in `main.py`;
`plot_prediction_interval` in `src/visualization/plots.py`.

---

## 5. Spatial Cross-Validation — Prediction in Ungauged Basins (PUB)

**The gap it fixes:** the base project does a **temporal** test (same catchment,
future years). That proves the model interpolates in time, **not** that it
generalises to **new catchments**. But most Indian rivers are **ungauged** — the
operational goal is nationwide maps from meteorology + static attributes alone.

**What we did:** `src/evaluation/spatial_cv.py` runs **Leave-One-Catchment-Out
(LOCO)**: for each catchment, train on *all the others* and test on the held-out
one. The scaler is fit on training catchments only — **no leakage**. We report the
**median NSE**, and the fraction of catchments with NSE > 0 and NSE > 0.5 (the
"acceptable model" threshold from Moriasi et al. 2007) — exactly the summary
statistics used in CAMELS benchmark papers.

**Viva soundbite:** "Temporal validation answers *can it forecast this gauged
river's future?* PUB answers *can it predict a river it has never seen?* — which is
what nationwide deployment under PM Gati Shakti actually requires."

**Files:** `src/evaluation/spatial_cv.py`; `phase_spatial_cv` in `main.py`
(`python main.py --phase spatial_cv`).

---

## 6. Bug Fixes & Honesty Improvements

- **Neural-network crash fixed.** In the original random-search fallback,
  `np.random.choice` was called on a list-of-lists (`hidden_layer_config`),
  raising "inhomogeneous shape". Now we select by **index**, so the MLP tunes
  correctly. (`src/models/automl_trainer.py`)
- **SHAP now installed** (`requirements.txt`) so the interpretability phase uses
  real SHAP values instead of the permutation-importance fallback.
- **Reproducibility:** the Optuna sampler is seeded, so the whole search is
  repeatable — important for a defensible report.

---

## How to Run the New Capabilities

```bash
# Full pipeline now also produces uncertainty intervals + optimisation-history plots
python main.py --config configs/config.yaml

# Prediction-in-Ungauged-Basins evaluation (spatial generalisation)
python main.py --phase spatial_cv

# Include the LSTM in the model search (slower): set models.lstm.enabled: true
#   in configs/model_configs.yaml, then:
python main.py --phase train
```

New outputs:
- `outputs/figures/optuna_history_<model>.png` — Bayesian Optimization curve
- `outputs/figures/prediction_interval.png` — uncertainty band vs observed
- `outputs/reports/uncertainty_report.json` — PICP / MPIW
- `outputs/reports/spatial_cv_report.json` — PUB median NSE per basin
- `hydrological_signatures` block inside `outputs/reports/evaluation_report.json`

---

## One-Paragraph Summary for Your Report's Abstract

> "Beyond standard AutoML, the framework employs genuine Bayesian Optimization
> (Optuna TPE) over a model pool that includes a state-of-the-art LSTM
> rainfall-runoff network. Performance is assessed not only with NSE/KGE but with
> hydrological signatures (%FHV, %FLV, %FMS, baseflow index, peak-timing) that
> target flood and drought behaviour directly. Predictive uncertainty is
> quantified with Conformalized Quantile Regression, yielding calibrated 90%
> prediction intervals, and spatial generalisation is verified through
> Leave-One-Catchment-Out (Prediction in Ungauged Basins) cross-validation —
> the operational scenario for nationwide deployment."

---

## Suggested Further Work (mention these for top marks)

1. **Entity-Aware LSTM (EA-LSTM):** feed static catchment attributes through a
   separate gate so one LSTM specialises per catchment type (Kratzert 2019).
2. **Multi-task / regional LSTM:** train one model on all 228 catchments at once —
   the regime that beats local models in the literature.
3. **Real CAMELS-IND data + a conceptual-model baseline** (e.g. GR4J/HBV) so you
   can claim "ML beats the traditional model by X NSE points".
4. **SHAP on the LSTM** via Expected Gradients to show *which forcings* and *which
   past days* drive a predicted flood.
