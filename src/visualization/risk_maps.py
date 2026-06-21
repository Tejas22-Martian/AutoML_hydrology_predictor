"""
Flood and drought risk map generation.

Classifies catchments into risk levels based on model predictions and generates
interactive and static maps for policymakers and disaster management agencies.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("streamflow_automl.visualization.risk_maps")


class RiskMapGenerator:
    """Generate catchment-level flood and drought risk maps."""

    RISK_COLORS = {
        "Very Low": "#2ecc71",
        "Low": "#82e0aa",
        "Moderate": "#f9e79f",
        "High": "#e74c3c",
        "Very High": "#922b21",
    }

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.risk_levels = self.config.get(
            "risk_levels", ["Very Low", "Low", "Moderate", "High", "Very High"]
        )
        self.output_dir = Path(self.config.get("output_dir", "outputs/risk_maps"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_risk_scores(
        self,
        model,
        X_data: np.ndarray,
        catchment_ids: list,
        catchment_attrs: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Compute flood and drought risk scores per catchment.

        Pass ``predictions`` to skip the internal ``model.predict`` (used by the
        per-catchment streaming path that predicts in small batches to bound
        memory over all 242 catchments). Otherwise predict on ``X_data``.
        """
        if predictions is None:
            predictions = model.predict(X_data)
        predictions = np.asarray(predictions)
        predictions = np.maximum(predictions, 0)

        results = []
        unique_catchments = np.unique(catchment_ids)

        for cid in unique_catchments:
            mask = np.array(catchment_ids) == cid
            catchment_preds = predictions[mask]

            if len(catchment_preds) < 30:
                continue

            flood_score = self._compute_flood_risk(catchment_preds)
            drought_score = self._compute_drought_risk(catchment_preds)
            combined_score = 0.5 * flood_score + 0.5 * drought_score

            result = {
                "catchment_id": cid,
                "flood_risk_score": round(flood_score, 4),
                "drought_risk_score": round(drought_score, 4),
                "combined_risk_score": round(combined_score, 4),
                "flood_risk_level": self._score_to_level(flood_score),
                "drought_risk_level": self._score_to_level(drought_score),
                "combined_risk_level": self._score_to_level(combined_score),
                "mean_predicted_flow": round(np.mean(catchment_preds), 3),
                "max_predicted_flow": round(np.max(catchment_preds), 3),
                "cv_predicted_flow": round(
                    np.std(catchment_preds) / (np.mean(catchment_preds) + 0.01), 4
                ),
            }

            if cid in catchment_attrs.index:
                attrs = catchment_attrs.loc[cid]
                for col in ["latitude", "longitude", "area_km2", "basin"]:
                    if col in attrs.index:
                        result[col] = attrs[col]

            results.append(result)

        risk_df = pd.DataFrame(results)
        logger.info(
            f"Risk scores computed for {len(risk_df)} catchments"
        )
        return risk_df

    def _compute_flood_risk(self, predictions: np.ndarray) -> float:
        """Compute flood risk score (0-1) based on extreme high flows."""
        p95 = np.percentile(predictions, 95)
        p99 = np.percentile(predictions, 99)
        mean_flow = np.mean(predictions)

        freq_extreme = np.mean(predictions > p95)
        magnitude = p99 / (mean_flow + 0.01)
        variability = np.std(predictions) / (mean_flow + 0.01)

        score = 0.4 * min(freq_extreme * 20, 1.0) + \
                0.4 * min(magnitude / 10, 1.0) + \
                0.2 * min(variability / 3, 1.0)
        return min(max(score, 0), 1)

    def _compute_drought_risk(self, predictions: np.ndarray) -> float:
        """Compute drought risk score (0-1) based on extreme low flows."""
        p5 = np.percentile(predictions, 5)
        p10 = np.percentile(predictions, 10)
        mean_flow = np.mean(predictions)

        low_flow_ratio = 1 - min(p5 / (mean_flow + 0.01), 1.0)
        freq_low = np.mean(predictions < p10)
        consecutive = self._max_consecutive_below(predictions, p10)
        duration_score = min(consecutive / 90, 1.0)

        score = 0.3 * low_flow_ratio + 0.3 * min(freq_low * 10, 1.0) + \
                0.4 * duration_score
        return min(max(score, 0), 1)

    @staticmethod
    def _max_consecutive_below(values: np.ndarray, threshold: float) -> int:
        """Find maximum consecutive days below a threshold."""
        below = values < threshold
        max_count = 0
        current = 0
        for b in below:
            if b:
                current += 1
                max_count = max(max_count, current)
            else:
                current = 0
        return max_count

    def _score_to_level(self, score: float) -> str:
        """Convert numeric risk score to categorical level."""
        quantiles = self.config.get("flood_risk_quantiles", [0.2, 0.4, 0.6, 0.8])
        for i, q in enumerate(quantiles):
            if score < q:
                return self.risk_levels[i]
        return self.risk_levels[-1]

    def generate_static_map(
        self,
        risk_df: pd.DataFrame,
        risk_type: str = "flood",
        save: bool = True,
    ) -> None:
        """Generate a static matplotlib risk map."""
        if "latitude" not in risk_df.columns or "longitude" not in risk_df.columns:
            logger.warning("Lat/lon not available, generating bar chart instead")
            self._plot_risk_bar_chart(risk_df, risk_type, save)
            return

        fig, ax = plt.subplots(figsize=(12, 10))

        score_col = f"{risk_type}_risk_score"
        level_col = f"{risk_type}_risk_level"

        scatter = ax.scatter(
            risk_df["longitude"],
            risk_df["latitude"],
            c=risk_df[score_col],
            cmap="RdYlGn_r",
            s=risk_df.get("area_km2", pd.Series([100] * len(risk_df))) / 20,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
            vmin=0,
            vmax=1,
        )

        cbar = plt.colorbar(scatter, ax=ax, label="Risk Score")
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.set_ticklabels(self.risk_levels + [""])

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(
            f"Catchment-Level {risk_type.title()} Risk Map - Peninsular India"
        )

        ax.set_xlim(72, 86)
        ax.set_ylim(8, 24)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            filepath = self.output_dir / f"{risk_type}_risk_map.png"
            plt.savefig(filepath, dpi=200, bbox_inches="tight")
            logger.info(f"Risk map saved to {filepath}")
        plt.close()

    def generate_interactive_map(self, risk_df: pd.DataFrame) -> None:
        """Generate an interactive Folium risk map (HTML)."""
        try:
            import folium
        except ImportError:
            logger.warning("folium not installed, skipping interactive map")
            return

        if "latitude" not in risk_df.columns:
            logger.warning("Lat/lon not available for interactive map")
            return

        center_lat = risk_df["latitude"].mean()
        center_lon = risk_df["longitude"].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        for _, row in risk_df.iterrows():
            color = self.RISK_COLORS.get(row.get("flood_risk_level", "Moderate"), "#f9e79f")
            popup_text = (
                f"<b>{row['catchment_id']}</b><br>"
                f"Basin: {row.get('basin', 'N/A')}<br>"
                f"Flood Risk: {row.get('flood_risk_level', 'N/A')} "
                f"({row.get('flood_risk_score', 0):.3f})<br>"
                f"Drought Risk: {row.get('drought_risk_level', 'N/A')} "
                f"({row.get('drought_risk_score', 0):.3f})<br>"
                f"Area: {row.get('area_km2', 'N/A')} km²"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup_text,
            ).add_to(m)

        filepath = self.output_dir / "interactive_risk_map.html"
        m.save(str(filepath))
        logger.info(f"Interactive map saved to {filepath}")

    def export_geojson(self, risk_df: pd.DataFrame) -> None:
        """Export risk data as GeoJSON for GIS integration."""
        if "latitude" not in risk_df.columns:
            logger.warning("Lat/lon not available for GeoJSON export")
            return

        features = []
        for _, row in risk_df.iterrows():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["longitude"]), float(row["latitude"])],
                },
                "properties": {
                    k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in row.items()
                    if k not in ("latitude", "longitude")
                },
            }
            features.append(feature)

        geojson = {"type": "FeatureCollection", "features": features}

        filepath = self.output_dir / "risk_data.geojson"
        with open(filepath, "w") as f:
            json.dump(geojson, f, indent=2, default=str)
        logger.info(f"GeoJSON exported to {filepath}")

    def generate_choropleth(
        self,
        risk_df: pd.DataFrame,
        shapefile: Optional[str],
        risk_type: str = "flood",
        save: bool = True,
    ) -> None:
        """Render a true catchment-polygon choropleth from the CAMELS-IND shapefile."""
        if not shapefile or not Path(shapefile).exists():
            logger.warning("Shapefile not available; skipping choropleth")
            return
        try:
            import geopandas as gpd
        except ImportError:
            logger.warning("geopandas not installed; skipping choropleth")
            return

        gdf = gpd.read_file(shapefile)
        wanted = {str(c) for c in risk_df["catchment_id"]}

        # Auto-detect the catchment-id field by overlap with our risk ids.
        id_col = None
        for col in gdf.columns:
            if col == "geometry":
                continue
            vals = {self._norm_id(v) for v in gdf[col]}
            if len(vals & wanted) >= max(1, len(wanted) // 2):
                id_col = col
                break
        if id_col is None:
            logger.warning("Could not match shapefile ids to catchments; skipping choropleth")
            return

        gdf["catchment_id"] = gdf[id_col].map(self._norm_id)
        score_col = f"{risk_type}_risk_score"
        merged = gdf.merge(
            risk_df[["catchment_id", score_col, f"{risk_type}_risk_level"]],
            on="catchment_id", how="inner",
        )
        if merged.empty:
            logger.warning("No catchments joined to shapefile; skipping choropleth")
            return

        fig, ax = plt.subplots(figsize=(12, 12))
        merged.plot(
            column=score_col, cmap="RdYlGn_r", vmin=0, vmax=1,
            legend=True, edgecolor="black", linewidth=0.3, ax=ax,
            legend_kwds={"label": f"{risk_type.title()} risk score", "shrink": 0.6},
        )
        ax.set_title(
            f"Catchment {risk_type.title()} Risk - CAMELS-IND ({len(merged)} catchments)"
        )
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        plt.tight_layout()
        if save:
            filepath = self.output_dir / f"{risk_type}_risk_choropleth.png"
            plt.savefig(filepath, dpi=200, bbox_inches="tight")
            logger.info(f"Choropleth saved to {filepath}")
        plt.close()

    @staticmethod
    def _norm_id(value) -> str:
        """Normalize a catchment id to a zero-stripped integer string when possible."""
        try:
            return str(int(float(value)))
        except (ValueError, TypeError):
            return str(value)

    def _plot_risk_bar_chart(
        self, risk_df: pd.DataFrame, risk_type: str, save: bool
    ) -> None:
        """Fallback bar chart when geographic coordinates are unavailable."""
        fig, ax = plt.subplots(figsize=(14, 6))

        score_col = f"{risk_type}_risk_score"
        sorted_df = risk_df.sort_values(score_col, ascending=False)

        colors = [
            self.RISK_COLORS[self._score_to_level(s)]
            for s in sorted_df[score_col]
        ]

        ax.bar(range(len(sorted_df)), sorted_df[score_col], color=colors)
        ax.set_xlabel("Catchment")
        ax.set_ylabel(f"{risk_type.title()} Risk Score")
        ax.set_title(f"Catchment {risk_type.title()} Risk Scores")

        plt.tight_layout()
        if save:
            filepath = self.output_dir / f"{risk_type}_risk_bar_chart.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

    def generate_summary_report(self, risk_df: pd.DataFrame) -> dict:
        """Generate a summary of risk distribution across catchments."""
        summary = {
            "total_catchments": len(risk_df),
            "flood_risk_distribution": {},
            "drought_risk_distribution": {},
        }

        for risk_type in ["flood", "drought"]:
            level_col = f"{risk_type}_risk_level"
            if level_col in risk_df.columns:
                dist = risk_df[level_col].value_counts().to_dict()
                summary[f"{risk_type}_risk_distribution"] = dist

        if "basin" in risk_df.columns:
            basin_summary = {}
            for basin, group in risk_df.groupby("basin"):
                basin_summary[basin] = {
                    "n_catchments": len(group),
                    "mean_flood_risk": round(group["flood_risk_score"].mean(), 4),
                    "mean_drought_risk": round(group["drought_risk_score"].mean(), 4),
                }
            summary["basin_summary"] = basin_summary

        filepath = self.output_dir / "risk_summary.json"
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Risk summary saved to {filepath}")
        return summary
