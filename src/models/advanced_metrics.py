"""
Advanced hydrological evaluation: signature-based metrics.

Standard metrics (NSE, KGE) summarise *average* performance but are dominated
by high flows because they square errors. Examiners and hydrologists want to
know whether the model reproduces specific, physically meaningful behaviours:
flood peaks, drought low-flows, and the overall shape of the flow regime.

These "hydrological signatures" follow Yilmaz et al. (2008) and the formulation
used throughout the CAMELS / LSTM benchmarking literature (Kratzert et al. 2019).
"""

import numpy as np


def log_nse(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 0.01) -> float:
    """
    Nash-Sutcliffe Efficiency on log-transformed flows.

    WHY: Ordinary NSE squares errors, so a 10 mm error on a 100 mm flood counts
    100x more than a 1 mm error on a 1 mm drought flow. Taking logs compresses
    the high flows and stretches the low flows, so log-NSE measures how well the
    model captures DROUGHT / low-flow behaviour. Reporting both NSE and log-NSE
    shows the model is good across the whole flow regime, not just at peaks.

    epsilon avoids log(0); a common choice is 1% of mean observed flow.
    """
    eps = epsilon * np.mean(y_true) if epsilon < 1 else epsilon
    log_true = np.log(y_true + eps)
    log_pred = np.log(np.maximum(y_pred, 0) + eps)
    num = np.sum((log_true - log_pred) ** 2)
    den = np.sum((log_true - np.mean(log_true)) ** 2)
    return 1.0 - num / den if den > 0 else 0.0


def fdc_high_flow_bias(y_true: np.ndarray, y_pred: np.ndarray,
                       high_quantile: float = 0.98) -> float:
    """
    %FHV - High-segment volume bias of the Flow Duration Curve.

    WHY: This is the FLOOD metric. It measures the percentage bias in the
    volume of the highest 2% of flows. A value near 0 means flood peaks are
    well reproduced; negative means the model systematically UNDER-predicts
    floods (dangerous for flood preparedness); positive means over-prediction.

    Defined over flows that are exceeded only `1 - high_quantile` of the time.
    """
    thr_true = np.quantile(y_true, high_quantile)
    high_true = y_true[y_true >= thr_true]
    # use the same temporal mask so we compare like-for-like timesteps
    mask = y_true >= thr_true
    high_true = y_true[mask]
    high_pred = y_pred[mask]
    denom = np.sum(high_true)
    if denom == 0:
        return 0.0
    return 100.0 * np.sum(high_pred - high_true) / denom


def fdc_low_flow_bias(y_true: np.ndarray, y_pred: np.ndarray,
                      low_quantile: float = 0.30, epsilon: float = 0.01) -> float:
    """
    %FLV - Low-segment volume bias of the Flow Duration Curve (log space).

    WHY: This is the DROUGHT metric. It measures bias in the lowest 30% of
    flows, computed on log-transformed values so small absolute differences in
    tiny flows are not ignored. Important for drought mitigation and minimum
    environmental-flow planning.
    """
    eps = epsilon * np.mean(y_true) if epsilon < 1 else epsilon
    thr = np.quantile(y_true, low_quantile)
    mask = y_true <= thr
    low_true = np.log(y_true[mask] + eps)
    low_pred = np.log(np.maximum(y_pred[mask], 0) + eps)
    # express relative to the minimum (anchor) as in Yilmaz et al. (2008)
    low_true = low_true - low_true.min()
    low_pred = low_pred - low_pred.min()
    denom = np.sum(low_true)
    if denom == 0:
        return 0.0
    return -100.0 * np.sum(low_pred - low_true) / denom


def fdc_mid_slope_bias(y_true: np.ndarray, y_pred: np.ndarray,
                       lower: float = 0.2, upper: float = 0.7,
                       epsilon: float = 0.01) -> float:
    """
    %FMS - Bias in the slope of the mid-segment of the Flow Duration Curve.

    WHY: The slope of the FDC between the 20th and 70th exceedance percentiles
    describes the *flashiness* / variability of the catchment - how quickly flow
    transitions from high to low. Getting this right means the model captures the
    rainfall-runoff dynamics, not just the mean. Relevant for both flood timing
    and water-resource reliability.
    """
    eps = epsilon * np.mean(y_true) if epsilon < 1 else epsilon

    def _slope(flows):
        sorted_desc = np.sort(flows)[::-1]
        n = len(sorted_desc)
        q_low = sorted_desc[min(int(lower * n), n - 1)]
        q_high = sorted_desc[min(int(upper * n), n - 1)]
        return np.log(q_low + eps) - np.log(q_high + eps)

    slope_true = _slope(y_true)
    slope_pred = _slope(np.maximum(y_pred, 0))
    if slope_true == 0:
        return 0.0
    return 100.0 * (slope_pred - slope_true) / slope_true


def baseflow_index(flows: np.ndarray, alpha: float = 0.925, passes: int = 3) -> float:
    """
    Baseflow Index (BFI) via the Lyne-Hollick recursive digital filter.

    WHY: BFI is the long-term ratio of baseflow (groundwater-fed slow flow) to
    total streamflow. It is a catchment "fingerprint". Comparing BFI of observed
    vs predicted series shows whether the model reproduces the right balance of
    quick storm runoff vs sustained baseflow - a deeper test than NSE.

    The Lyne-Hollick filter separates quick flow from baseflow by repeatedly
    applying a one-parameter low-pass filter (alpha ~ 0.9-0.95 for daily data).
    """
    q = np.asarray(flows, dtype=float)
    q = np.maximum(q, 0)
    baseflow = q.copy()
    for p in range(passes):
        qf = np.zeros_like(q)
        # forward on odd passes, backward on even, as recommended
        seq = range(1, len(q)) if p % 2 == 0 else range(len(q) - 2, -1, -1)
        prev = 0.0
        for i in seq:
            j = i - 1 if p % 2 == 0 else i + 1
            qf[i] = alpha * qf[j] + (1 + alpha) / 2 * (baseflow[i] - baseflow[j])
            qf[i] = max(qf[i], 0)
            baseflow[i] = min(baseflow[i], q[i] - qf[i]) if qf[i] < q[i] else baseflow[i]
        baseflow = np.minimum(baseflow, q)
    total = np.sum(q)
    return float(np.sum(baseflow) / total) if total > 0 else 0.0


def runoff_ratio(flows: np.ndarray, precip: np.ndarray) -> float:
    """
    Runoff Ratio = mean(streamflow) / mean(precipitation).

    WHY: A first-order water-balance check. It tells you what fraction of rain
    becomes streamflow (the rest evaporates or recharges deep groundwater).
    A physically implausible runoff ratio (>1, or near 0) signals the model or
    data is wrong. Cheap, powerful sanity check for a report.
    """
    mp = np.mean(precip)
    return float(np.mean(flows) / mp) if mp > 0 else 0.0


def peak_timing_error(y_true: np.ndarray, y_pred: np.ndarray,
                      high_quantile: float = 0.95, window: int = 3) -> float:
    """
    Fraction of observed flood peaks the model captures within +/- `window` days.

    WHY: For flood EARLY WARNING, getting the *timing* right matters as much as
    the magnitude. This counts how many observed peaks (above the 95th percentile)
    have a predicted peak nearby. Reported as a hit-rate in [0, 1].
    """
    thr = np.quantile(y_true, high_quantile)
    peak_idx = np.where(y_true >= thr)[0]
    if len(peak_idx) == 0:
        return 0.0
    hits = 0
    for i in peak_idx:
        lo, hi = max(0, i - window), min(len(y_pred), i + window + 1)
        if np.max(y_pred[lo:hi]) >= thr:
            hits += 1
    return hits / len(peak_idx)


def compute_signatures(y_true: np.ndarray, y_pred: np.ndarray,
                       precip: np.ndarray = None) -> dict:
    """Compute the full suite of advanced signatures at once."""
    sig = {
        "log_nse": round(log_nse(y_true, y_pred), 4),
        "fhv_high_flow_bias_pct": round(fdc_high_flow_bias(y_true, y_pred), 2),
        "flv_low_flow_bias_pct": round(fdc_low_flow_bias(y_true, y_pred), 2),
        "fms_mid_slope_bias_pct": round(fdc_mid_slope_bias(y_true, y_pred), 2),
        "bfi_observed": round(baseflow_index(y_true), 4),
        "bfi_predicted": round(baseflow_index(y_pred), 4),
        "peak_timing_hit_rate": round(peak_timing_error(y_true, y_pred), 4),
    }
    if precip is not None:
        sig["runoff_ratio_observed"] = round(runoff_ratio(y_true, precip), 4)
        sig["runoff_ratio_predicted"] = round(runoff_ratio(y_pred, precip), 4)
    return sig
