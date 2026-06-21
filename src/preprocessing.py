"""
Data preprocessing module for the Astram Traffic Intelligence Platform.
Handles: datetime parsing, coordinate cleaning, missing value imputation,
duration calculation, and duplicate removal.

Fixes applied:
- Issue 3.1: Added `coord_imputed` flag column for imputed coordinates.
- Issue 3.2: Added `duration_imputed` flag column so regressor can exclude imputed rows.
- Issue 3.3: Logs warning for unrecognized `requires_road_closure` values before fillna.
- Issue 3.4: Filters end timestamps per-row to ignore values before start_datetime.
- Issue 5.1: Replaced hardcoded absolute paths with pathlib relative paths.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load raw event data, clean it, handle missing values, calculate durations,
    and drop duplicates.

    Parameters:
        filepath (str): Path to the raw CSV file.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame with `coord_imputed`
                       and `duration_imputed` flag columns.
    """
    logger.info(f"Loading raw data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # ── 1. Clean column names ────────────────────────────────────────────────
    df.columns = [col.strip() for col in df.columns]

    # ── 2. Convert datetime columns to timezone-naive datetimes ──────────────
    datetime_cols = [
        'start_datetime', 'end_datetime', 'resolved_datetime',
        'closed_datetime', 'modified_datetime', 'created_date',
    ]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)

    # ── 3. Handle requires_road_closure (Fix 3.3: log unknowns) ──────────────
    if 'requires_road_closure' in df.columns:
        raw_vals = df['requires_road_closure'].astype(str).str.upper().str.strip()
        known_map = {
            'TRUE': True, 'FALSE': False,
            'YES': True, 'NO': False,
            '1': True, '0': False,
            'NAN': False, 'NONE': False, 'NULL': False,
        }
        mapped = raw_vals.map(known_map)
        n_unknown = mapped.isna().sum()
        if n_unknown > 0:
            unknown_vals = raw_vals[mapped.isna()].unique().tolist()
            logger.warning(
                f"requires_road_closure: {n_unknown} rows had unrecognized values "
                f"{unknown_vals}. Defaulting these to False."
            )
        df['requires_road_closure'] = mapped.fillna(False).astype(bool)
    else:
        df['requires_road_closure'] = False

    # ── 4. Categorical / string missing values ───────────────────────────────
    df['priority'] = df['priority'].fillna('Medium').astype(str).str.strip().str.capitalize()
    df['event_cause'] = df['event_cause'].fillna('others').astype(str).str.strip().str.lower()
    df['event_type'] = df['event_type'].fillna('unplanned').astype(str).str.strip().str.lower()

    for cat_col in ['zone', 'junction', 'police_station', 'corridor']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna('Unknown').astype(str).str.strip()
            # Treat literal "NULL" strings as Unknown
            df.loc[df[cat_col].str.upper() == 'NULL', cat_col] = 'Unknown'

    # ── 5. Coordinate cleaning and imputation (Fix 3.1: add flag) ────────────
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Mark out-of-bounds or missing coordinates
    invalid_coords = (
        (df['latitude'] < 12.0) | (df['latitude'] > 14.0) |
        (df['longitude'] < 77.0) | (df['longitude'] > 78.5) |
        df['latitude'].isna() | df['longitude'].isna()
    )
    df.loc[invalid_coords, ['latitude', 'longitude']] = np.nan

    # Flag rows that need coordinate imputation BEFORE imputing
    df['coord_imputed'] = df['latitude'].isna() | df['longitude'].isna()
    n_coord_imputed = df['coord_imputed'].sum()
    logger.info(f"Coordinate imputation: {n_coord_imputed} rows flagged as coord_imputed=True.")

    # Impute: police station mean → zone mean → global median
    police_coords = df.groupby('police_station')[['latitude', 'longitude']].transform('mean')
    df['latitude'] = df['latitude'].fillna(police_coords['latitude'])
    df['longitude'] = df['longitude'].fillna(police_coords['longitude'])

    zone_coords = df.groupby('zone')[['latitude', 'longitude']].transform('mean')
    df['latitude'] = df['latitude'].fillna(zone_coords['latitude'])
    df['longitude'] = df['longitude'].fillna(zone_coords['longitude'])

    global_lat_median = df['latitude'].median()
    global_lon_median = df['longitude'].median()
    df['latitude'] = df['latitude'].fillna(global_lat_median)
    df['longitude'] = df['longitude'].fillna(global_lon_median)

    # ── 6. Duration calculation (Fixes 3.4, 3.2) ────────────────────────────
    # Fix 3.4: per-row, ignore end timestamps that precede start_datetime
    end_candidates = ['end_datetime', 'resolved_datetime', 'closed_datetime']
    existing_end_cols = [c for c in end_candidates if c in df.columns]

    def _earliest_valid_end(row):
        """Return the earliest end timestamp that is >= start_datetime."""
        start = row['start_datetime']
        if pd.isna(start):
            return pd.NaT
        candidates = []
        for col in existing_end_cols:
            val = row[col]
            if pd.notna(val) and val >= start:
                candidates.append(val)
        return min(candidates) if candidates else pd.NaT

    min_end_time = df.apply(_earliest_valid_end, axis=1)
    df['duration_minutes'] = (min_end_time - df['start_datetime']).dt.total_seconds() / 60.0

    # Cap extreme durations (>30 days = 43200 min)
    df.loc[
        (df['duration_minutes'] < 0) | (df['duration_minutes'] > 43200),
        'duration_minutes'
    ] = np.nan

    # Fix 3.2: flag rows whose duration is missing BEFORE imputation
    df['duration_imputed'] = df['duration_minutes'].isna()
    n_dur_imputed = df['duration_imputed'].sum()
    logger.info(f"Duration imputation: {n_dur_imputed} rows flagged as duration_imputed=True.")

    # Impute: cause+priority median → cause median → global median
    median_durations = df.groupby(['event_cause', 'priority'])['duration_minutes'].transform('median')
    df['duration_minutes'] = df['duration_minutes'].fillna(median_durations)

    cause_medians = df.groupby('event_cause')['duration_minutes'].transform('median')
    df['duration_minutes'] = df['duration_minutes'].fillna(cause_medians)

    global_median_duration = df['duration_minutes'].median()
    if pd.isna(global_median_duration):
        global_median_duration = 60.0
    df['duration_minutes'] = df['duration_minutes'].fillna(global_median_duration)

    # ── 7. Remove duplicate records ──────────────────────────────────────────
    initial_rows = len(df)
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='first')
    else:
        df = df.drop_duplicates(keep='first')
    duplicate_count = initial_rows - len(df)
    logger.info(f"Removed {duplicate_count} duplicate records. Remaining: {len(df)} rows.")

    return df


# Fix 5.1: use relative paths instead of hardcoded absolute paths
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    test_file = project_root / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if test_file.exists():
        processed_df = preprocess_data(str(test_file))
        print(processed_df[['id', 'event_type', 'event_cause', 'latitude', 'longitude',
                            'duration_minutes', 'coord_imputed', 'duration_imputed']].head(10))
    else:
        print(f"Test file not found at {test_file}, skipping verification.")
