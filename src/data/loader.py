"""
Data loader for the real CAMELS-IND dataset (FOSEE release).

Reads the dataset *natively* in its published layout:

    <fosee_root>/
        catchment_mean_forcings/{id:05d}.csv      # per-catchment daily forcings
        streamflow_timeseries/streamflow_observed.csv   # wide: year,month,day,<id>...
        streamflow_timeseries/lstm_pred_streamflow.csv  # regional-LSTM baseline
        attributes_csv/camels_ind_*.csv           # static attributes keyed by gauge_id
        shapefiles_catchment/catchments.shp       # catchment polygons

Observed/predicted streamflow is stored in m3/s and is converted here to
mm/day using the catchment drainage area (``cwc_area``):

    q[mm/day] = q[m3/s] * 86.4 / area[km2]

No synthetic data is generated anywhere in this project.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("streamflow_automl.data.loader")

# Raw forcing header (lower-cased) -> canonical column name used downstream.
# pet/aet use the complete GLEAM series (the bare ``pet(mm/day)`` column has gaps).
FORCING_COLUMN_MAP = {
    "prcp(mm/day)": "precipitation_mm",
    "tmax(c)": "temperature_max_c",
    "tmin(c)": "temperature_min_c",
    "tavg(c)": "temperature_avg_c",
    "srad_lw(w/m2)": "solar_radiation_lw_wm2",
    "srad_sw(w/m2)": "solar_radiation_wm2",
    "wind(m/s)": "wind_speed_ms",
    "rel_hum(%)": "relative_humidity_pct",
    "pet_gleam(mm/day)": "pet_mm",
    "aet_gleam(mm/day)": "aet_mm",
    "sm_lvl1(kg/m2)": "soil_moisture_l1",
    "sm_lvl2(kg/m2)": "soil_moisture_l2",
    "sm_lvl3(kg/m2)": "soil_moisture_l3",
    "sm_lvl4(kg/m2)": "soil_moisture_l4",
}

# Source attribute column -> canonical name (others are kept verbatim).
ATTRIBUTE_RENAME = {
    "cwc_area": "area_km2",
    "cwc_lat": "latitude",
    "cwc_lon": "longitude",
    "river_basin": "basin",
    "aridity_p_pet": "aridity_index",
    "p_monthly_variability": "p_seasonality",
    "pop_density_2020": "pop_density",
    "urban_frac_2005": "urban_frac",
}

SECONDS_PER_DAY = 86400.0


def normalize_id(catchment_id) -> str:
    """Canonical catchment id: zero-stripped integer string (e.g. '3002')."""
    return str(int(catchment_id))


class CAMELSIndDataLoader:
    """Loads and merges real CAMELS-IND forcings, streamflow, and attributes."""

    def __init__(self, data_config: dict, attribute_features: Optional[list] = None):
        self.config = data_config or {}
        self.forcings_dir = Path(self.config["forcings_dir"])
        self.streamflow_file = Path(self.config["streamflow_file"])
        self.attributes_dir = Path(self.config["attributes_dir"])
        lstm_pred = self.config.get("lstm_pred_file")
        self.lstm_pred_file = Path(lstm_pred) if lstm_pred else None
        # Static attributes attached to each row as model features. Restricted to an
        # explicit list so streamflow-derived (leakage) attributes are never included.
        self.attribute_features = (
            attribute_features
            if attribute_features is not None
            else self.config.get("catchment_attributes", [])
        )

        self._attributes: Optional[pd.DataFrame] = None
        self._observed: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Attributes
    # ------------------------------------------------------------------ #
    def get_attributes(self) -> pd.DataFrame:
        """Merge all attribute CSVs on gauge_id; index by canonical catchment id."""
        if self._attributes is not None:
            return self._attributes

        files = sorted(self.attributes_dir.glob("camels_ind_*.csv"))
        if not files:
            raise FileNotFoundError(f"No attribute CSVs in {self.attributes_dir}")

        merged: Optional[pd.DataFrame] = None
        for f in files:
            df = pd.read_csv(f)
            if "gauge_id" not in df.columns:
                continue
            if merged is None:
                merged = df
            else:
                dup = [c for c in df.columns if c in merged.columns and c != "gauge_id"]
                df = df.drop(columns=dup)
                merged = merged.merge(df, on="gauge_id", how="outer")

        merged = merged.rename(columns=ATTRIBUTE_RENAME).copy()
        merged["catchment_id"] = merged["gauge_id"].map(normalize_id)
        merged = merged.set_index("catchment_id")
        self._attributes = merged
        logger.info(
            f"Loaded attributes: {len(merged)} catchments, {merged.shape[1]} columns"
        )
        return merged

    # ------------------------------------------------------------------ #
    # Catchment listing / basin filtering
    # ------------------------------------------------------------------ #
    def list_catchments(self) -> list:
        """Catchment ids present in forcings, streamflow, and attributes."""
        if not self.forcings_dir.exists():
            return []
        forcing_ids = {normalize_id(f.stem) for f in self.forcings_dir.glob("*.csv")}
        obs_ids = {
            c for c in self._observed_streamflow().columns
            if c not in ("year", "month", "day", "date")
        }
        attr_ids = set(self.get_attributes().index)
        return sorted(forcing_ids & obs_ids & attr_ids, key=int)

    def get_basin_catchments(self, basins) -> list:
        """Catchment ids whose river_basin is in ``basins`` (case-insensitive)."""
        if isinstance(basins, str):
            basins = [basins]
        wanted = {b.lower() for b in basins}
        attrs = self.get_attributes()
        mask = attrs["basin"].str.lower().isin(wanted)
        ids = set(attrs.index[mask])
        return sorted(ids & set(self.list_catchments()), key=int)

    def filter_trainable(
        self, catchment_ids: list, min_samples: int, start: str, end: str
    ) -> list:
        """Keep catchments with >= ``min_samples`` valid observed days in [start, end]."""
        obs = self._observed_streamflow()
        window = (obs["date"] >= pd.Timestamp(start)) & (obs["date"] <= pd.Timestamp(end))
        sub = obs.loc[window]
        kept = [
            c for c in catchment_ids
            if c in sub.columns and sub[c].notna().sum() >= min_samples
        ]
        dropped = len(catchment_ids) - len(kept)
        if dropped:
            logger.info(f"Filtered out {dropped} catchments below {min_samples} train samples")
        return kept

    # ------------------------------------------------------------------ #
    # Streamflow (wide -> per-catchment, m3/s -> mm/day)
    # ------------------------------------------------------------------ #
    def _observed_streamflow(self) -> pd.DataFrame:
        if self._observed is None:
            self._observed = self._read_wide_streamflow(self.streamflow_file)
        return self._observed

    @staticmethod
    def _read_wide_streamflow(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path).copy()
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df.columns = [
            normalize_id(c) if c not in ("year", "month", "day", "date") else c
            for c in df.columns
        ]
        return df

    def _area_km2(self, catchment_id: str) -> float:
        area = self.get_attributes().loc[catchment_id, "area_km2"]
        if pd.isna(area) or area <= 0:
            raise ValueError(f"Invalid drainage area for catchment {catchment_id}: {area}")
        return float(area)

    @staticmethod
    def cms_to_mm_day(q_cms: pd.Series, area_km2: float) -> pd.Series:
        """Convert discharge in m3/s to area-normalized runoff depth in mm/day."""
        return q_cms * SECONDS_PER_DAY / (area_km2 * 1e6) * 1000.0

    # ------------------------------------------------------------------ #
    # Forcings
    # ------------------------------------------------------------------ #
    def _load_forcings(self, catchment_id: str) -> pd.DataFrame:
        path = self.forcings_dir / f"{int(catchment_id):05d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Forcing file not found: {path}")

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df[["year", "month", "day"]])
        df = df.rename(
            columns={c: FORCING_COLUMN_MAP.get(c.strip().lower(), c) for c in df.columns}
        )

        keep = ["date"] + [c for c in FORCING_COLUMN_MAP.values() if c in df.columns]
        df = df[keep].sort_values("date").reset_index(drop=True)

        # Fill the small forcing gaps within this single catchment (no cross-catchment bleed).
        numeric = df.columns.drop("date")
        df[numeric] = df[numeric].interpolate(method="linear").bfill().ffill()
        return df

    # ------------------------------------------------------------------ #
    # Single / multi catchment assembly
    # ------------------------------------------------------------------ #
    def load_catchment(self, catchment_id: str) -> pd.DataFrame:
        """Load one catchment: forcings + observed target (mm/day) + static attrs."""
        cid = normalize_id(catchment_id)
        forcings = self._load_forcings(cid)

        obs = self._observed_streamflow()
        if cid not in obs.columns:
            raise FileNotFoundError(f"No observed streamflow column for catchment {cid}")
        area = self._area_km2(cid)
        target = pd.DataFrame({
            "date": obs["date"],
            "streamflow_mm_day": self.cms_to_mm_day(obs[cid], area),
        })

        merged = forcings.merge(target, on="date", how="inner")

        attrs = self.get_attributes().loc[cid]
        for col in self.attribute_features:
            if col in attrs.index:
                merged[col] = attrs[col]
        # Always carry geo/identity columns for mapping and grouping.
        merged["catchment_id"] = cid
        merged["basin"] = attrs.get("basin")
        merged["latitude"] = attrs.get("latitude")
        merged["longitude"] = attrs.get("longitude")
        return merged

    def load_all_catchments(self, catchment_ids: Optional[list] = None) -> pd.DataFrame:
        """Load and stack multiple catchments into one long DataFrame."""
        if catchment_ids is None:
            catchment_ids = self.list_catchments()

        frames = []
        for cid in catchment_ids:
            try:
                frames.append(self.load_catchment(cid))
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Skipping catchment {cid}: {e}")

        if not frames:
            raise ValueError("No catchment data loaded successfully")
        data = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(catchment_ids)} catchments -> {len(data)} rows")
        return data

    # ------------------------------------------------------------------ #
    # Regional-LSTM baseline (for benchmarking)
    # ------------------------------------------------------------------ #
    def load_lstm_predictions(self, catchment_ids: Optional[list] = None) -> pd.DataFrame:
        """Long-format regional-LSTM predictions in mm/day: [date, catchment_id, lstm_mm_day]."""
        if self.lstm_pred_file is None or not self.lstm_pred_file.exists():
            logger.warning("LSTM prediction file not available")
            return pd.DataFrame(columns=["date", "catchment_id", "lstm_mm_day"])

        wide = self._read_wide_streamflow(self.lstm_pred_file)
        if catchment_ids is None:
            catchment_ids = [
                c for c in wide.columns if c not in ("year", "month", "day", "date")
            ]

        attrs = self.get_attributes()
        records = []
        for cid in catchment_ids:
            if cid not in wide.columns or cid not in attrs.index:
                continue
            area = self._area_km2(cid)
            records.append(pd.DataFrame({
                "date": wide["date"],
                "catchment_id": cid,
                "lstm_mm_day": self.cms_to_mm_day(wide[cid], area),
            }))
        if not records:
            return pd.DataFrame(columns=["date", "catchment_id", "lstm_mm_day"])
        return pd.concat(records, ignore_index=True)
