"""
Hydrological visualization utilities.

Generates standard hydrological plots: hydrographs, scatter plots, flow duration
curves, residual analysis, and seasonal performance comparisons.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.metrics import compute_all_metrics

logger = logging.getLogger("streamflow_automl.visualization.plots")


class HydrologicalPlotter:
    """Generate publication-quality hydrological analysis plots."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = {
            "observed": "#1f77b4",
            "predicted": "#ff7f0e",
            "flood": "#d62728",
            "drought": "#8c564b",
        }

    def plot_hydrograph(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Streamflow Hydrograph",
        save: bool = True,
    ) -> None:
        """Plot observed vs predicted streamflow time series."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

        ax1.plot(dates, y_true, label="Observed", color=self.colors["observed"],
                 linewidth=0.8, alpha=0.9)
        ax1.plot(dates, y_pred, label="Predicted", color=self.colors["predicted"],
                 linewidth=0.8, alpha=0.7)
        ax1.set_ylabel("Streamflow (mm/day)")
        ax1.set_title(title)
        ax1.legend(loc="upper right")

        metrics = compute_all_metrics(y_true, y_pred)
        metrics_text = f"NSE={metrics['nse']:.3f}  KGE={metrics['kge']:.3f}  RMSE={metrics['rmse']:.3f}"
        ax1.text(0.02, 0.95, metrics_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        residuals = y_true - y_pred
        ax2.bar(dates, residuals, color="gray", alpha=0.5, width=1)
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.set_ylabel("Residual")
        ax2.set_xlabel("Date")

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "hydrograph.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Observed vs Predicted",
        save: bool = True,
    ) -> None:
        """Scatter plot of observed vs predicted with 1:1 line."""
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=self.colors["observed"])

        lims = [
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="1:1 Line")

        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=1,
                label=f"Regression (slope={z[0]:.3f})")

        metrics = compute_all_metrics(y_true, y_pred)
        ax.text(0.05, 0.95,
                f"NSE = {metrics['nse']:.4f}\nKGE = {metrics['kge']:.4f}\n"
                f"R² = {metrics['r2']:.4f}\nPBIAS = {metrics['pbias']:.2f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))

        ax.set_xlabel("Observed Streamflow (mm/day)")
        ax.set_ylabel("Predicted Streamflow (mm/day)")
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "scatter_plot.png", dpi=150, bbox_inches="tight")
        plt.close()

    def plot_flow_duration_curve(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = True,
    ) -> None:
        """Plot flow duration curves for observed and predicted."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for data, label, color in [
            (y_true, "Observed", self.colors["observed"]),
            (y_pred, "Predicted", self.colors["predicted"]),
        ]:
            sorted_data = np.sort(data)[::-1]
            exceedance = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            ax.semilogy(exceedance, sorted_data, label=label, color=color, linewidth=1.5)

        ax.set_xlabel("Exceedance Probability (%)")
        ax.set_ylabel("Streamflow (mm/day)")
        ax.set_title("Flow Duration Curve")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "flow_duration_curve.png", dpi=150,
                        bbox_inches="tight")
        plt.close()

    def plot_monthly_performance(
        self,
        dates: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = True,
    ) -> None:
        """Boxplot of monthly NSE values."""
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "observed": y_true,
            "predicted": y_pred,
        })
        df["month"] = df["date"].dt.month

        monthly_nse = []
        for month in range(1, 13):
            mask = df["month"] == month
            if mask.sum() > 10:
                metrics = compute_all_metrics(
                    df.loc[mask, "observed"].values,
                    df.loc[mask, "predicted"].values,
                )
                monthly_nse.append({"month": month, "nse": metrics["nse"]})

        nse_df = pd.DataFrame(monthly_nse)

        fig, ax = plt.subplots(figsize=(10, 5))
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        colors = ["#e74c3c" if m in [6, 7, 8, 9] else "#3498db" for m in range(1, 13)]

        bars = ax.bar(nse_df["month"], nse_df["nse"],
                      color=[colors[m - 1] for m in nse_df["month"]])
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        ax.set_ylabel("Nash-Sutcliffe Efficiency")
        ax.set_title("Monthly Model Performance (Red = Monsoon)")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "monthly_performance.png", dpi=150,
                        bbox_inches="tight")
        plt.close()

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        save: bool = True,
    ) -> None:
        """Radar/bar chart comparing multiple models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        metrics_to_plot = ["nse", "kge", "r2"]
        available = [m for m in metrics_to_plot if m in comparison_df.columns]
        comparison_df[available].plot(kind="bar", ax=axes[0], colormap="viridis")
        axes[0].set_title("Model Comparison (Higher is Better)")
        axes[0].set_ylabel("Score")
        axes[0].legend(title="Metric")
        axes[0].tick_params(axis="x", rotation=45)

        if "rmse" in comparison_df.columns:
            comparison_df["rmse"].plot(kind="bar", ax=axes[1], color="#e74c3c")
            axes[1].set_title("RMSE Comparison (Lower is Better)")
            axes[1].set_ylabel("RMSE (mm/day)")
            axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "model_comparison.png", dpi=150,
                        bbox_inches="tight")
        plt.close()

    def plot_optimization_history(self, study, model_name: str = "model",
                                  save: bool = True) -> None:
        """
        Plot the Optuna optimisation curve: best NSE found vs trial number.

        A rising-then-flattening curve is the visual proof that Bayesian
        Optimization is *learning* - it finds better configs early and converges,
        unlike random search which improves only by luck. Great figure for the
        methodology section.
        """
        trials = [t for t in study.trials if t.value is not None]
        if not trials:
            return
        values = [t.value for t in trials]
        running_best = np.maximum.accumulate(values)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(range(len(values)), values, "o", alpha=0.4,
                color="#95a5a6", label="Trial NSE")
        ax.plot(range(len(running_best)), running_best, "-",
                color="#2980b9", linewidth=2, label="Best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("CV Nash-Sutcliffe Efficiency")
        ax.set_title(f"Bayesian Optimization History - {model_name}")
        ax.legend()
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f"optuna_history_{model_name}.png",
                        dpi=150, bbox_inches="tight")
        plt.close()

    def plot_prediction_interval(self, dates, y_true, lower, median, upper,
                                 picp: float = None, max_points: int = 400,
                                 save: bool = True) -> None:
        """
        Plot the prediction interval (shaded band) against observed flow.

        Shows where the model is confident (narrow band) vs uncertain (wide band).
        If most observed points fall inside the band, the uncertainty estimate is
        well-calibrated - annotate with the observed coverage (PICP).
        """
        n = min(len(y_true), max_points)
        x = np.arange(n)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.fill_between(x, lower[:n], upper[:n], color="#3498db", alpha=0.25,
                        label="Prediction interval")
        ax.plot(x, median[:n], color="#2980b9", linewidth=1, label="Median forecast")
        ax.plot(x, y_true[:n], color="black", linewidth=0.8, label="Observed")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Streamflow (mm/day)")
        title = "Streamflow Prediction Interval"
        if picp is not None:
            title += f"  (observed coverage = {picp * 100:.1f}%)"
        ax.set_title(title)
        ax.legend(loc="upper right")
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / "prediction_interval.png", dpi=150,
                        bbox_inches="tight")
        plt.close()

    def plot_benchmark(self, summary: dict, output_name: str = "benchmark",
                       save: bool = True) -> None:
        """Plot per-catchment NSE: AutoML vs the regional-LSTM baseline.

        Left: scatter of paired NSE (points above the 1:1 line = AutoML wins).
        Right: distribution of NSE for both models.
        """
        per = pd.DataFrame(summary.get("per_catchment", []))
        if per.empty:
            logger.warning("No benchmark data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        lim_lo = float(min(per["automl_nse"].min(), per["lstm_nse"].min(), 0))
        lim_lo = max(lim_lo, -1.0)
        ax1.plot([lim_lo, 1], [lim_lo, 1], "k--", linewidth=1, label="1:1 line")
        ax1.scatter(per["lstm_nse"].clip(lower=-1), per["automl_nse"].clip(lower=-1),
                    c="#2980b9", alpha=0.6, edgecolors="black", linewidth=0.3)
        ax1.set_xlim(lim_lo, 1); ax1.set_ylim(lim_lo, 1)
        ax1.set_xlabel("Regional-LSTM NSE")
        ax1.set_ylabel("AutoML NSE")
        ax1.set_title(
            f"Per-catchment NSE (AutoML wins {summary.get('automl_win_rate_nse', 0)*100:.0f}%)"
        )
        ax1.legend(loc="lower right"); ax1.grid(True, alpha=0.3)

        data = [per["automl_nse"].clip(lower=-1), per["lstm_nse"].clip(lower=-1)]
        # matplotlib >=3.9 renamed boxplot's `labels` -> `tick_labels`.
        ax2.boxplot(data, showmeans=True)
        ax2.set_xticklabels(["AutoML", "Regional-LSTM"])
        ax2.set_ylabel("NSE (clipped at -1)")
        ax2.set_title("NSE distribution across catchments")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / f"{output_name}.png", dpi=150,
                        bbox_inches="tight")
        plt.close()
