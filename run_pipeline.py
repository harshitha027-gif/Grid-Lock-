"""
Pipeline runner for the Astram Traffic Intelligence Platform.
Orchestrates: preprocessing → risk scoring → model training → resource
recommendation → CSV export.

Fixes applied:
- Issue 5.1: Replaced hardcoded absolute paths with pathlib-based relative paths.
- Issue 5.4: Saves model_metadata.json with training date, row count, sklearn version.
- Issue 1.3: Risk scores computed on fully-featured data; train/test split now
             happens inside train_models() BEFORE feature engineering.
"""

import os
import sys
import json
import datetime
from pathlib import Path

import pandas as pd
import sklearn
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.risk_engine import train_models, calculate_risk_score
from src.resource_recommender import recommend_resources_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Event-Driven Congestion Forecasting Pipeline...")

    # ── 1. Define paths (Fix 5.1: relative) ──────────────────────────────────
    raw_data_path = PROJECT_ROOT / "Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"

    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    try:
        processed_df = preprocess_data(str(raw_data_path))
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    # ── 3. Train ML models ───────────────────────────────────────────────────
    #   train_models() now internally:
    #     - Filters out imputed-duration rows (Fix 1.1)
    #     - Splits BEFORE feature engineering (Fix 1.3)
    #     - Returns the full-featured df for downstream use
    logger.info("Training predictive machine learning models...")
    try:
        metrics, featured_df = train_models(processed_df, save_dir=str(models_dir))
        logger.info("Model training completed successfully.")
        logger.info(
            f"Regressor MAE: {metrics['regressor_mae']:.2f} mins, "
            f"R² Score: {metrics['regressor_r2']:.4f}"
        )
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)

    # ── 4. Compute risk scores on the full featured dataset ──────────────────
    logger.info("Computing risk scores and levels...")
    try:
        featured_df['risk_score'], featured_df['risk_level'] = calculate_risk_score(featured_df)
        logger.info(f"Risk levels distribution:\n{featured_df['risk_level'].value_counts()}")
    except Exception as e:
        logger.error(f"Risk scoring failed: {e}")
        sys.exit(1)

    # ── 5. Resource recommendations ──────────────────────────────────────────
    try:
        final_df = recommend_resources_df(featured_df)
    except Exception as e:
        logger.error(f"Resource recommendation failed: {e}")
        sys.exit(1)

    # ── 6. Save outputs ──────────────────────────────────────────────────────
    try:
        output_csv = processed_dir / "processed_traffic_events.csv"
        final_df.to_csv(str(output_csv), index=False)
        logger.info(f"Saved processed dataset to {output_csv}")

        coords_csv = processed_dir / "historical_coords.csv"
        coords_df = final_df[['latitude', 'longitude', 'junction', 'zone']].drop_duplicates()
        coords_df.to_csv(str(coords_csv), index=False)
        logger.info(f"Saved coordinate lookup file to {coords_csv}")
    except Exception as e:
        logger.error(f"Saving outputs failed: {e}")
        sys.exit(1)

    # ── 7. Save model metadata (Fix 5.4) ─────────────────────────────────────
    metadata = {
        'trained_at': datetime.datetime.now().isoformat(),
        'dataset_rows': len(processed_df),
        'sklearn_version': sklearn.__version__,
        'metrics': metrics,
    }
    metrics_path = models_dir / "model_metrics.json"
    with open(str(metrics_path), 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to {metrics_path}")

    logger.info("Pipeline executed successfully!")


if __name__ == '__main__':
    main()
