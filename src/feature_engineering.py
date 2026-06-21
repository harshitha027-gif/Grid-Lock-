"""
Feature engineering module for the Astram Traffic Intelligence Platform.
Computes temporal features (hour, day, weekend), zone/junction frequencies,
and geospatial event density via Haversine BallTree.

Fixes applied:
- Issue 5.1: Replaced hardcoded absolute path with pathlib relative path.
- Note on Issue 1.3: The is_training / historical_df parameter already supports
  leak-free usage. The fix for 1.3 is in HOW this function is called (see
  risk_engine.py and run_pipeline.py).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import logging

from src.config import DENSITY_RADIUS_KM, EARTH_RADIUS_KM

logger = logging.getLogger(__name__)


def engineer_features(
    df: pd.DataFrame,
    is_training: bool = True,
    historical_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generate temporal and spatial features.

    Parameters:
        df: The DataFrame to enrich with features.
        is_training: If True, computes frequencies/densities from df itself.
                     If False, uses historical_df as the reference.
        historical_df: Reference data for inference mode (frequencies & density).

    Returns:
        DataFrame with new columns: hour_of_day, day_of_week, is_weekend,
        zone_frequency, junction_frequency, historical_event_density.
    """
    df = df.copy()
    logger.info("Engineering temporal features...")

    # ── 1. Temporal Features ─────────────────────────────────────────────────
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # ── 2. Reference DataFrame for frequency / density ───────────────────────
    ref_df = df if is_training else (historical_df if historical_df is not None else df)

    logger.info("Engineering spatial frequency features...")

    # Zone Frequency
    zone_counts = ref_df['zone'].value_counts()
    df['zone_frequency'] = df['zone'].map(zone_counts).fillna(0).astype(int)

    # Junction Frequency
    junction_counts = ref_df['junction'].value_counts()
    df['junction_frequency'] = df['junction'].map(junction_counts).fillna(0).astype(int)

    logger.info("Engineering geospatial density features...")

    # ── 3. Historical Density (Haversine BallTree) ───────────────────────────
    r_rad = DENSITY_RADIUS_KM / EARTH_RADIUS_KM   # radius in radians

    ref_coords = np.radians(ref_df[['latitude', 'longitude']].values)
    query_coords = np.radians(df[['latitude', 'longitude']].values)

    tree = BallTree(ref_coords, metric='haversine')
    counts = tree.query_radius(query_coords, r=r_rad, count_only=True)
    df['historical_event_density'] = counts

    logger.info(
        "Feature engineering completed. Columns added: hour_of_day, "
        "day_of_week, is_weekend, zone_frequency, junction_frequency, "
        "historical_event_density"
    )
    return df


# Fix 5.1: relative paths
if __name__ == '__main__':
    from src.preprocessing import preprocess_data

    project_root = Path(__file__).resolve().parent.parent
    test_file = project_root / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if test_file.exists():
        processed_df = preprocess_data(str(test_file))
        featured_df = engineer_features(processed_df)
        print(featured_df[['id', 'hour_of_day', 'day_of_week', 'is_weekend',
                           'zone_frequency', 'junction_frequency',
                           'historical_event_density']].head())
    else:
        print(f"Test file not found at {test_file}, skipping verification.")
