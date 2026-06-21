"""
Uncertainty quantification via quantile regression.

WHY THIS EARNS MARKS
--------------------
A single number ("tomorrow's flow = 5 mm") is not enough for risk management.
A decision-maker needs "we are 90% confident the flow is between 3 and 9 mm".
Point predictions hide uncertainty; for FLOOD warnings and DROUGHT planning the
*upper and lower bounds* are what trigger action. Adding prediction intervals
turns the model from a forecaster into a risk tool - and it gives you two more
rigorous evaluation metrics (PICP and MPIW) for the report.

HOW QUANTILE REGRESSION WORKS (viva answer)
-------------------------------------------
Ordinary regression minimises squared error and predicts the conditional MEAN.
Quantile regression instead minimises the "pinball" (quantile) loss, which is
asymmetric: for the 0.95 quantile, under-predictions are penalised 0.95 and
over-predictions only 0.05, so the model is pushed up until ~95% of observations
fall below it. Fitting models at the 0.05 and 0.95 quantiles yields a 90%
prediction interval directly, with no Gaussian assumption.

CONFORMALIZED QUANTILE REGRESSION (the upgrade, Romano et al. 2019)
------------------------------------------------------------------
Plain quantile regression often UNDER-covers on unseen data (the model is a bit
overconfident). CQR fixes this with a calibration step that gives a *finite-
sample coverage guarantee* with no distributional assumptions:
  1. hold out a calibration set the quantile models never trained on,
  2. measure how far each true value falls OUTSIDE the predicted band
     (the "conformity score" E = max(lo - y, y - hi)),
  3. take the (1-alpha) quantile of those scores, Q,
  4. widen every interval by Q: [lo - Q, hi + Q].
The result provably contains the truth at least (1-alpha) of the time. This turns
a heuristic interval into a statistically rigorous one - a strong report point.
"""

import logging

import numpy as np

logger = logging.getLogger("streamflow_automl.models.uncertainty")


class QuantileIntervalEstimator:
    """Fits lower / median / upper quantile models to build prediction intervals."""

    def __init__(self, lower: float = 0.05, upper: float = 0.95,
                 n_estimators: int = 300, max_depth: int = 4,
                 learning_rate: float = 0.05, conformalize: bool = True,
                 calibration_fraction: float = 0.2, non_negative: bool = True,
                 random_state: int = 42):
        self.lower = lower
        self.upper = upper
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.conformalize = conformalize
        self.calibration_fraction = calibration_fraction
        self.non_negative = non_negative  # clamp to >=0 (streamflow is non-negative)
        self.random_state = random_state
        self.models = {}
        self.conformal_q = 0.0  # interval widening from the calibration step

    def fit(self, X, y):
        """Fit lower / median / upper quantile models, then conformal-calibrate."""
        from sklearn.ensemble import GradientBoostingRegressor

        X = np.asarray(X)
        y = np.asarray(y)

        # split off a calibration set the quantile models never see (for CQR)
        if self.conformalize and len(y) > 50:
            n_cal = max(20, int(self.calibration_fraction * len(y)))
            X_fit, y_fit = X[:-n_cal], y[:-n_cal]
            X_cal, y_cal = X[-n_cal:], y[-n_cal:]
        else:
            X_fit, y_fit, X_cal, y_cal = X, y, None, None

        for name, q in [("lower", self.lower), ("median", 0.5), ("upper", self.upper)]:
            model = GradientBoostingRegressor(
                loss="quantile", alpha=q,
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, random_state=self.random_state,
            )
            model.fit(X_fit, y_fit)
            self.models[name] = model

        if X_cal is not None:
            lo = self.models["lower"].predict(X_cal)
            hi = self.models["upper"].predict(X_cal)
            # conformity score: how far outside the band each truth lies
            # (negative when the truth is comfortably inside the interval).
            scores = np.maximum(lo - y_cal, y_cal - hi)
            n = len(scores)
            target_coverage = self.upper - self.lower
            # CQR: take the ceil((n+1)*coverage)/n-th smallest score (Romano 2019).
            k = int(np.ceil((n + 1) * target_coverage))
            k = min(max(k, 1), n)
            self.conformal_q = float(np.sort(scores)[k - 1])

        logger.info(
            f"Fitted quantile interval [{self.lower}, {self.upper}] "
            f"(nominal {100 * (self.upper - self.lower):.0f}%); "
            f"conformal widening Q={self.conformal_q:.3f}"
        )
        return self

    def predict_interval(self, X):
        """Return (lower, median, upper) arrays; conformal-widened and ordered."""
        lo = self.models["lower"].predict(X) - self.conformal_q
        med = self.models["median"].predict(X)
        hi = self.models["upper"].predict(X) + self.conformal_q
        # quantile crossing can occasionally invert bounds; sort to fix
        stacked = np.sort(np.vstack([lo, med, hi]), axis=0)
        lo, med, hi = stacked[0], stacked[1], stacked[2]
        if self.non_negative:
            lo, med, hi = np.maximum(lo, 0), np.maximum(med, 0), np.maximum(hi, 0)
        return lo, med, hi

    def evaluate_interval(self, X, y_true) -> dict:
        """
        Compute interval-quality metrics.

        PICP - Prediction Interval Coverage Probability: the fraction of true
               values that actually fall inside the interval. Should be close to
               the nominal level (e.g. ~0.90 for a 90% interval). Too low = over-
               confident; too high = unnecessarily wide.
        MPIW - Mean Prediction Interval Width: average upper-minus-lower. Narrower
               is better *provided* PICP stays near nominal. Captures the
               sharpness/usefulness of the interval.
        """
        lo, med, hi = self.predict_interval(X)
        inside = (y_true >= lo) & (y_true <= hi)
        picp = float(np.mean(inside))
        mpiw = float(np.mean(hi - lo))
        nominal = self.upper - self.lower
        return {
            "nominal_coverage": round(nominal, 3),
            "picp_observed_coverage": round(picp, 4),
            "mpiw_mean_interval_width": round(mpiw, 4),
            "coverage_error": round(picp - nominal, 4),
        }
