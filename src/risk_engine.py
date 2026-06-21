"""
Risk scoring engine and ML model training for the Astram Traffic Intelligence Platform.

Fixes applied:
- Issue 1.1: Duration regressor trained only on non-imputed rows (duration_imputed==False).
- Issue 1.2: ML risk classifier REMOVED. Risk level derived solely from deterministic formula.
- Issue 1.3: Train/test split happens BEFORE feature engineering to prevent data leakage.
- Issue 2.1: Scoring formula uses shared constants from config.py (no duplication).
- Issue 2.2: predict_event() derives risk_level from formula score (no ML classifier).
- Issue 2.3: zone/junction frequencies computed from historical data at inference, not hardcoded.
- Issue 5.3: Vectorized cause score calculation (no more iterrows).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import BallTree
import joblib
import logging

from src.config import (
    RISK_WEIGHTS, PRIORITY_SCORE_MAP, CAUSE_SCORE_MAP,
    CAUSE_SCORE_DEFAULT, UNPLANNED_BONUS, DURATION_MAX_MINUTES,
    RISK_THRESHOLDS, DENSITY_RADIUS_KM, EARTH_RADIUS_KM,
)

logger = logging.getLogger(__name__)


# ─── Shared scoring helpers (used everywhere) ────────────────────────────────

def _compute_cause_scores(df: pd.DataFrame) -> pd.Series:
    """Vectorized cause score computation (Fix 5.3: no iterrows)."""
    base = df['event_cause'].map(CAUSE_SCORE_MAP).fillna(CAUSE_SCORE_DEFAULT)
    bonus = np.where(df['event_type'] == 'unplanned', UNPLANNED_BONUS, 0.0)
    return (base + bonus).clip(upper=100.0)


def _score_to_level(score: float) -> str:
    """Map a numeric risk score to a categorical level using shared thresholds."""
    for threshold, level in RISK_THRESHOLDS:
        if score < threshold:
            return level
    return 'Critical'  # fallback


def calculate_risk_score(df: pd.DataFrame) -> tuple:
    """
    Compute numeric risk scores (0-100) and categorical risk levels.
    Uses shared constants from config.py — the ONLY place scoring logic lives.

    Parameters:
        df: DataFrame with columns: requires_road_closure, priority, event_cause,
            event_type, duration_minutes, historical_event_density.

    Returns:
        (pd.Series of scores, pd.Series of level labels)
    """
    w = RISK_WEIGHTS

    # Component scores
    road_closure_score = df['requires_road_closure'].apply(lambda x: 100.0 if x else 0.0)
    priority_score = df['priority'].map(PRIORITY_SCORE_MAP).fillna(60.0)
    cause_score = _compute_cause_scores(df)
    duration_score = (df['duration_minutes'] / DURATION_MAX_MINUTES * 100.0).clip(upper=100.0).fillna(30.0)

    p95_density = df['historical_event_density'].quantile(0.95)
    if pd.isna(p95_density) or p95_density == 0:
        p95_density = 10.0
    density_score = (df['historical_event_density'] / p95_density * 100.0).clip(upper=100.0)

    total = (
        w['road_closure'] * road_closure_score +
        w['priority']     * priority_score +
        w['cause']        * cause_score +
        w['duration']     * duration_score +
        w['density']      * density_score
    )

    levels = total.apply(_score_to_level)
    return total, levels


def compute_factor_breakdown(
    requires_road_closure: bool,
    priority: str,
    event_cause: str,
    event_type: str,
    duration_minutes: float,
    historical_event_density: float,
    p95_density: float,
) -> dict:
    """
    Return a dict of individual factor scores for UI decomposition display.
    Uses the same constants as calculate_risk_score().
    """
    road_closure_s = 100.0 if requires_road_closure else 0.0
    priority_s = PRIORITY_SCORE_MAP.get(priority, 60.0)

    base_cause = CAUSE_SCORE_MAP.get(event_cause.lower().strip(), CAUSE_SCORE_DEFAULT)
    if event_type == 'unplanned':
        base_cause = min(100.0, base_cause + UNPLANNED_BONUS)
    cause_s = base_cause

    duration_s = min(100.0, (duration_minutes / DURATION_MAX_MINUTES) * 100.0)

    if p95_density == 0:
        p95_density = 10.0
    density_s = min(100.0, (historical_event_density / p95_density) * 100.0)

    return {
        'road_closure_score': round(road_closure_s, 1),
        'priority_score':     round(priority_s, 1),
        'cause_score':        round(cause_s, 1),
        'duration_score':     round(duration_s, 1),
        'density_score':      round(density_s, 1),
    }


# ─── Model training ─────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, save_dir: str = 'models') -> dict:
    """
    Train and save the Duration Regressor.
    The ML risk classifier has been REMOVED (Fix 1.2) — risk levels come
    solely from the deterministic scoring formula.

    Fix 1.1: Trains only on rows where duration_imputed == False.
    Fix 1.3: Feature engineering frequencies/densities are computed on the
             training split only, then mapped onto the test split.

    Parameters:
        df: Preprocessed DataFrame (must contain duration_imputed column).
        save_dir: Directory to persist model artifacts.

    Returns:
        dict with regressor performance metrics.
    """
    from src.feature_engineering import engineer_features

    os.makedirs(save_dir, exist_ok=True)
    df = df.copy()

    # ── Fix 1.1: exclude imputed-duration rows from regressor training ───────
    if 'duration_imputed' in df.columns:
        real_dur_df = df[df['duration_imputed'] == False].copy()
        logger.info(
            f"Duration regressor: using {len(real_dur_df)}/{len(df)} rows "
            f"with real (non-imputed) durations."
        )
    else:
        real_dur_df = df.copy()
        logger.warning("No duration_imputed column found — using all rows.")

    # ── Fix 1.3: split BEFORE feature engineering ────────────────────────────
    train_df, test_df = train_test_split(
        real_dur_df, test_size=0.2, random_state=42,
    )

    # Engineer features: training set uses itself as reference
    train_df = engineer_features(train_df, is_training=True)
    # Test set uses training set as reference (no leakage)
    test_df = engineer_features(test_df, is_training=False, historical_df=train_df)

    # ── Prepare features & target ────────────────────────────────────────────
    feature_cols = [
        'event_type', 'event_cause', 'priority', 'requires_road_closure',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'zone_frequency', 'junction_frequency', 'historical_event_density',
    ]
    categorical_features = ['event_type', 'event_cause', 'priority']
    numeric_features = [
        'requires_road_closure', 'hour_of_day', 'day_of_week', 'is_weekend',
        'zone_frequency', 'junction_frequency', 'historical_event_density',
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['duration_minutes']
    X_test = test_df[feature_cols]
    y_test = test_df['duration_minutes']

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features),
    ])

    # ── Train Duration Regressor ─────────────────────────────────────────────
    logger.info("Training Duration Regressor model...")
    regressor_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=150, random_state=42)),
    ])
    regressor_pipeline.fit(X_train, y_train)
    y_pred = regressor_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"Regressor MAE: {mae:.2f} minutes, R² Score: {r2:.4f}")

    # ── Persist artifacts ────────────────────────────────────────────────────
    joblib.dump(regressor_pipeline, os.path.join(save_dir, 'duration_regressor.joblib'))

    # Also compute and save risk scores on the FULL featured dataset for CSV export
    # (features computed on full df for the output file — acceptable for batch export)
    full_featured = engineer_features(df, is_training=True)
    p95_density = full_featured['historical_event_density'].quantile(0.95)
    joblib.dump(float(p95_density), os.path.join(save_dir, 'p95_density.joblib'))

    # Save zone/junction frequency maps for inference (Fix 2.3)
    zone_freq_map = full_featured['zone'].value_counts().to_dict()
    junction_freq_map = full_featured['junction'].value_counts().to_dict()
    joblib.dump(zone_freq_map, os.path.join(save_dir, 'zone_freq_map.joblib'))
    joblib.dump(junction_freq_map, os.path.join(save_dir, 'junction_freq_map.joblib'))

    # Remove old classifier if it exists (Fix 1.2)
    old_classifier = os.path.join(save_dir, 'risk_classifier.joblib')
    if os.path.exists(old_classifier):
        os.remove(old_classifier)
        logger.info("Removed legacy risk_classifier.joblib (no longer used).")

    metrics = {
        'regressor_mae': float(mae),
        'regressor_r2': float(r2),
        'training_rows': len(train_df),
        'test_rows': len(test_df),
    }
    return metrics, full_featured


# ─── Single-event inference ──────────────────────────────────────────────────

def predict_event(
    model_dir: str,
    event_data: dict,
    historical_df: pd.DataFrame = None,
) -> dict:
    """
    Predict disruption duration and compute risk score for a single event.

    Fix 2.2: risk_level is derived from the formula score (no ML classifier).
    Fix 2.3: zone/junction frequencies loaded from saved maps, not hardcoded.

    Parameters:
        model_dir: Directory containing saved model artifacts.
        event_data: Dict with keys: event_type, event_cause, priority,
                    requires_road_closure, start_datetime, latitude, longitude,
                    zone, junction.
        historical_df: Full historical DataFrame for density calculation.

    Returns:
        dict with risk_score, risk_level, predicted_duration_minutes,
        and factor_breakdown.
    """
    # Load artifacts
    regressor = joblib.load(os.path.join(model_dir, 'duration_regressor.joblib'))
    p95_density = joblib.load(os.path.join(model_dir, 'p95_density.joblib'))
    zone_freq_map = joblib.load(os.path.join(model_dir, 'zone_freq_map.joblib'))
    junction_freq_map = joblib.load(os.path.join(model_dir, 'junction_freq_map.joblib'))

    # Build single-row DataFrame
    df = pd.DataFrame([event_data])
    df['start_datetime'] = pd.to_datetime(df['start_datetime']).dt.tz_localize(None)
    df['requires_road_closure'] = df['requires_road_closure'].astype(bool)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)

    # Temporal features
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Fix 2.3: real frequency lookups from saved maps
    zone_val = str(df['zone'].iloc[0])
    junction_val = str(df['junction'].iloc[0])
    df['zone_frequency'] = zone_freq_map.get(zone_val, 0)
    df['junction_frequency'] = junction_freq_map.get(junction_val, 0)

    # Spatial density
    if historical_df is not None and not historical_df.empty:
        r_rad = DENSITY_RADIUS_KM / EARTH_RADIUS_KM
        ref_coords = np.radians(historical_df[['latitude', 'longitude']].values)
        query_coords = np.radians(df[['latitude', 'longitude']].values)
        tree = BallTree(ref_coords, metric='haversine')
        density = tree.query_radius(query_coords, r=r_rad, count_only=True)[0]
        df['historical_event_density'] = int(density)
    else:
        df['historical_event_density'] = 1

    # Predict duration
    feature_cols = [
        'event_type', 'event_cause', 'priority', 'requires_road_closure',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'zone_frequency', 'junction_frequency', 'historical_event_density',
    ]
    predicted_duration = float(regressor.predict(df[feature_cols])[0])

    # Fix 2.2: derive risk_level from formula score, NOT from an ML classifier
    breakdown = compute_factor_breakdown(
        requires_road_closure=bool(event_data['requires_road_closure']),
        priority=event_data['priority'],
        event_cause=event_data['event_cause'],
        event_type=event_data['event_type'],
        duration_minutes=predicted_duration,
        historical_event_density=int(df['historical_event_density'].iloc[0]),
        p95_density=p95_density,
    )

    w = RISK_WEIGHTS
    risk_score = (
        w['road_closure'] * breakdown['road_closure_score'] +
        w['priority']     * breakdown['priority_score'] +
        w['cause']        * breakdown['cause_score'] +
        w['duration']     * breakdown['duration_score'] +
        w['density']      * breakdown['density_score']
    )
    risk_level = _score_to_level(risk_score)

    return {
        'risk_score': round(float(risk_score), 1),
        'risk_level': risk_level,
        'predicted_duration_minutes': round(predicted_duration, 1),
        'factor_breakdown': breakdown,
    }
