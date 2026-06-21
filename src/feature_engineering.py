import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import logging

logger = logging.getLogger(__name__)

def engineer_features(df: pd.DataFrame, is_training: bool = True, historical_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate temporal and spatial features for the dataset.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to extract features for.
    is_training (bool): If True, computes frequencies and densities from df itself.
                        If False, uses the provided historical_df to calculate frequency/density features.
    historical_df (pd.DataFrame): Historical data reference for inference mode.
    
    Returns:
    pd.DataFrame: DataFrame with engineered features.
    """
    df = df.copy()
    logger.info("Engineering temporal features...")
    
    # 1. Temporal Features
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 2. Reference DataFrame for Frequency and Density
    ref_df = df if is_training else (historical_df if historical_df is not None else df)
    
    logger.info("Engineering spatial frequency features...")
    
    # Zone Frequency
    zone_counts = ref_df['zone'].value_counts()
    df['zone_frequency'] = df['zone'].map(zone_counts).fillna(0).astype(int)
    
    # Junction Frequency
    junction_counts = ref_df['junction'].value_counts()
    df['junction_frequency'] = df['junction'].map(junction_counts).fillna(0).astype(int)
    
    logger.info("Engineering geospatial density features...")
    
    # Historical Density (500m radius using Haversine BallTree)
    # Earth radius = 6371.0 km. 500m = 0.5 km.
    # Radius in radians = 0.5 / 6371.0
    r_rad = 0.5 / 6371.0
    
    # Get reference coordinates
    ref_coords = np.radians(ref_df[['latitude', 'longitude']].values)
    query_coords = np.radians(df[['latitude', 'longitude']].values)
    
    # Build BallTree on reference coordinates
    tree = BallTree(ref_coords, metric='haversine')
    
    # Query density
    counts = tree.query_radius(query_coords, r=r_rad, count_only=True)
    
    df['historical_event_density'] = counts
    
    logger.info(f"Feature engineering completed. Columns added: hour_of_day, day_of_week, is_weekend, zone_frequency, junction_frequency, historical_event_density")
    return df

if __name__ == '__main__':
    # Simple verification test
    from preprocessing import preprocess_data
    import os
    
    test_file = r"c:\Users\Harshitha\Desktop\Grid lock project\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if os.path.exists(test_file):
        processed_df = preprocess_data(test_file)
        featured_df = engineer_features(processed_df)
        print(featured_df[['id', 'hour_of_day', 'day_of_week', 'is_weekend', 'zone_frequency', 'junction_frequency', 'historical_event_density']].head())
    else:
        print("Test file not found, skipping verification.")
