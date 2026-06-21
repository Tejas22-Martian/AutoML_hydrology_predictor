"""
Hydrological evaluation metrics.

Standard metrics used in hydrology for evaluating streamflow predictions:
NSE, KGE, RMSE, PBIAS, and R².
"""

import numpy as np


def nash_sutcliffe_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency (NSE).

    Range: -inf to 1.0. NSE=1 is perfect, NSE=0 equals mean prediction,
    NSE<0 is worse than predicting the mean.
    """
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def kling_gupta_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency (KGE).

    Decomposes into correlation, variability bias, and mean bias.
    Range: -inf to 1.0. KGE=1 is perfect.
    """
    r = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 else 0.0
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else 0.0
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else 0.0
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error (RMSE). Lower is better."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def percent_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percent Bias (PBIAS).

    Measures average tendency of predictions to be larger/smaller than observed.
    Optimal value is 0. Positive = underestimation, Negative = overestimation.
    """
    total_obs = np.sum(y_true)
    if total_obs == 0:
        return 0.0
    return 100.0 * np.sum(y_true - y_pred) / total_obs


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²). Range: 0 to 1."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all hydrological metrics at once."""
    return {
        "nse": round(nash_sutcliffe_efficiency(y_true, y_pred), 4),
        "kge": round(kling_gupta_efficiency(y_true, y_pred), 4),
        "rmse": round(root_mean_square_error(y_true, y_pred), 4),
        "pbias": round(percent_bias(y_true, y_pred), 4),
        "r2": round(r_squared(y_true, y_pred), 4),
    }


def compute_extreme_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    flood_percentile: float = 95,
    drought_percentile: float = 5,
) -> dict:
    """Compute metrics specifically for extreme events (floods/droughts)."""
    flood_threshold = np.percentile(y_true, flood_percentile)
    drought_threshold = np.percentile(y_true, drought_percentile)

    flood_mask = y_true >= flood_threshold
    drought_mask = y_true <= drought_threshold

    results = {"overall": compute_all_metrics(y_true, y_pred)}

    if np.sum(flood_mask) > 5:
        results["flood_events"] = compute_all_metrics(
            y_true[flood_mask], y_pred[flood_mask]
        )
        results["flood_events"]["n_events"] = int(np.sum(flood_mask))

    if np.sum(drought_mask) > 5:
        results["drought_events"] = compute_all_metrics(
            y_true[drought_mask], y_pred[drought_mask]
        )
        results["drought_events"]["n_events"] = int(np.sum(drought_mask))

    return results
