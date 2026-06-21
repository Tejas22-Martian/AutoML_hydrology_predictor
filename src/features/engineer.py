"""
Feature engineering for hydrological time series.

Creates lag features, rolling statistics, seasonal encodings, and integrates
static catchment attributes to build a rich feature set for streamflow prediction.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("streamflow_automl.features.engineer")


class FeatureEngineer:
    """Creates predictive features from hydrometeorological time series."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.lag_days = self.config.get("lag_days", [1, 2, 3, 5, 7, 14, 30])
        self.rolling_windows = self.config.get("rolling_windows", [3, 7, 14, 30])
        self.rolling_stats = self.config.get(
            "rolling_statistics", ["mean", "std", "min", "max"]
        )
        self.met_features = self.config.get("meteorological_features", [
            "precipitation_mm", "temperature_max_c", "temperature_min_c",
            "relative_humidity_pct", "wind_speed_ms", "solar_radiation_wm2", "pet_mm",
        ])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps, independently per catchment.

        Lag/rolling/derived features use ``shift``/``rolling``/``diff`` which must
        never cross catchment boundaries — so the temporal operations are applied
        within each ``catchment_id`` group, then the groups are concatenated.
        """
        data = data.copy()
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"])

        if "catchment_id" in data.columns:
            groups = [
                self._transform_single(g)
                for _, g in data.groupby("catchment_id", sort=False)
            ]
            data = pd.concat(groups, ignore_index=True)
        else:
            data = self._transform_single(data)

        # Global cleanup: static attributes are constant within a catchment but vary
        # across them, so constant/empty-column pruning runs on the pooled frame.
        initial_cols = len(data.columns)
        data = data.dropna(axis=1, how="all")
        keep = [c for c in data.columns if c in ("date", "catchment_id", "basin")
                or data[c].nunique(dropna=True) > 1]
        data = data[keep]
        final_cols = len(data.columns)

        if initial_cols != final_cols:
            logger.info(f"Removed {initial_cols - final_cols} constant/empty columns")

        logger.info(
            f"Feature engineering complete: {final_cols} columns, {len(data)} samples"
        )
        return data

    def _transform_single(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the sequential feature steps on a single catchment's time series."""
        data = data.sort_values("date").reset_index(drop=True)
        data = self._add_lag_features(data)
        data = self._add_rolling_features(data)
        data = self._add_seasonal_features(data)
        data = self._add_derived_features(data)

        max_lag = max(self.lag_days + self.rolling_windows)
        return data.iloc[max_lag:].reset_index(drop=True)

    def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of key variables."""
        target = "streamflow_mm_day"
        if target in data.columns:
            for lag in self.lag_days:
                data[f"{target}_lag_{lag}"] = data[target].shift(lag)

        for col in self.met_features:
            if col not in data.columns:
                continue
            for lag in [1, 3, 7]:
                data[f"{col}_lag_{lag}"] = data[col].shift(lag)

        return data

    def _add_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistics."""
        target = "streamflow_mm_day"
        precip = "precipitation_mm"

        for col in [target, precip]:
            if col not in data.columns:
                continue
            for window in self.rolling_windows:
                rolling = data[col].rolling(window=window, min_periods=1)
                for stat in self.rolling_stats:
                    data[f"{col}_roll{window}_{stat}"] = getattr(rolling, stat)()

        if precip in data.columns:
            for window in [3, 7, 14, 30]:
                data[f"{precip}_cumsum_{window}"] = (
                    data[precip].rolling(window=window, min_periods=1).sum()
                )
            data[f"{precip}_dry_days_7"] = (
                (data[precip] < 0.1)
                .rolling(window=7, min_periods=1)
                .sum()
            )

        return data

    def _add_seasonal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical seasonal encodings and monsoon indicators."""
        if "date" not in data.columns:
            return data

        dates = pd.to_datetime(data["date"])
        doy = dates.dt.dayofyear

        encode_method = self.config.get("seasonal", {}).get(
            "encode_method", "cyclical"
        )

        if encode_method == "cyclical":
            data["day_sin"] = np.sin(2 * np.pi * doy / 365.25)
            data["day_cos"] = np.cos(2 * np.pi * doy / 365.25)
            month = dates.dt.month
            data["month_sin"] = np.sin(2 * np.pi * month / 12)
            data["month_cos"] = np.cos(2 * np.pi * month / 12)
        else:
            data["day_of_year"] = doy
            data["month"] = dates.dt.month

        month = dates.dt.month
        data["is_monsoon"] = month.isin([6, 7, 8, 9]).astype(int)
        data["is_pre_monsoon"] = month.isin([3, 4, 5]).astype(int)
        data["is_post_monsoon"] = month.isin([10, 11]).astype(int)
        data["is_winter"] = month.isin([12, 1, 2]).astype(int)

        return data

    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create physically meaningful derived features."""
        if "temperature_max_c" in data.columns and "temperature_min_c" in data.columns:
            data["temperature_range_c"] = (
                data["temperature_max_c"] - data["temperature_min_c"]
            )
            data["temperature_mean_c"] = (
                data["temperature_max_c"] + data["temperature_min_c"]
            ) / 2

        if "precipitation_mm" in data.columns and "pet_mm" in data.columns:
            data["precip_pet_ratio"] = data["precipitation_mm"] / (
                data["pet_mm"] + 0.01
            )
            data["water_balance_mm"] = data["precipitation_mm"] - data["pet_mm"]

        if "precipitation_mm" in data.columns:
            data["precip_intensity"] = np.where(
                data["precipitation_mm"] > 0, data["precipitation_mm"], 0
            )
            data["is_rainy_day"] = (data["precipitation_mm"] > 1.0).astype(int)

        if "streamflow_mm_day" in data.columns:
            data["streamflow_change"] = data["streamflow_mm_day"].diff()
            data["streamflow_pct_change"] = data["streamflow_mm_day"].pct_change(
                fill_method=None
            )
            data["streamflow_pct_change"] = data["streamflow_pct_change"].clip(-10, 10)

        return data

    def get_feature_names(self, data: pd.DataFrame) -> list:
        """Return names of all engineered features (excluding target, date, IDs)."""
        exclude = {
            "streamflow_mm_day", "date", "catchment_id", "basin",
            "latitude", "longitude",
        }
        return [
            c for c in data.columns
            if c not in exclude and data[c].dtype in [np.float64, np.float32, np.int64]
        ]
