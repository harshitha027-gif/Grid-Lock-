import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Load raw event data, clean it, handle missing values, calculate durations, and drop duplicates.
    
    Parameters:
    filepath (str): Path to the raw CSV file.
    
    Returns:
    pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    logger.info(f"Loading raw data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    
    # 1. Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # 2. Convert datetime columns to timezone-naive datetimes
    datetime_cols = ['start_datetime', 'end_datetime', 'resolved_datetime', 'closed_datetime', 'modified_datetime', 'created_date']
    for col in datetime_cols:
        if col in df.columns:
            # Parse dates and localize to UTC then convert to naive to prevent timezone mismatch issues
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.tz_localize(None)
            
    # 3. Handle requires_road_closure conversions
    if 'requires_road_closure' in df.columns:
        df['requires_road_closure'] = df['requires_road_closure'].astype(str).str.upper().str.strip()
        df['requires_road_closure'] = df['requires_road_closure'].map(
            {'TRUE': True, 'FALSE': False, 'YES': True, 'NO': False, '1': True, '0': False, 'NAN': False}
        ).fillna(False).astype(bool)
    else:
        df['requires_road_closure'] = False
        
    # 4. Handle other categorical and string missing values
    df['priority'] = df['priority'].fillna('Medium').astype(str).str.strip().str.capitalize()
    df['event_cause'] = df['event_cause'].fillna('others').astype(str).str.strip().str.lower()
    df['event_type'] = df['event_type'].fillna('unplanned').astype(str).str.strip().str.lower()
    
    for cat_col in ['zone', 'junction', 'police_station', 'corridor']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].fillna('Unknown').astype(str).str.strip()
            
    # 5. Coordinate cleaning and imputation
    # Parse to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    
    # Mark coordinates that are out of bounds or zero as NaN
    # Bengaluru coordinates roughly: Lat 12.5 - 13.5, Lon 77.2 - 77.9
    invalid_coords = (df['latitude'] < 12.0) | (df['latitude'] > 14.0) | \
                     (df['longitude'] < 77.0) | (df['longitude'] > 78.5) | \
                     df['latitude'].isna() | df['longitude'].isna()
    df.loc[invalid_coords, ['latitude', 'longitude']] = np.nan
    
    # Impute missing coordinates using police station mean coordinates
    police_coords = df.groupby('police_station')[['latitude', 'longitude']].transform('mean')
    df['latitude'] = df['latitude'].fillna(police_coords['latitude'])
    df['longitude'] = df['longitude'].fillna(police_coords['longitude'])
    
    # If still missing, use zone mean coordinates
    zone_coords = df.groupby('zone')[['latitude', 'longitude']].transform('mean')
    df['latitude'] = df['latitude'].fillna(zone_coords['latitude'])
    df['longitude'] = df['longitude'].fillna(zone_coords['longitude'])
    
    # If still missing, fill with global median
    global_lat_median = df['latitude'].median()
    global_lon_median = df['longitude'].median()
    df['latitude'] = df['latitude'].fillna(global_lat_median)
    df['longitude'] = df['longitude'].fillna(global_lon_median)
    
    # 6. Calculate event duration in minutes
    # We find the minimum valid end timestamp (from end_datetime, resolved_datetime, closed_datetime)
    # that is greater than start_datetime.
    end_candidates = ['end_datetime', 'resolved_datetime', 'closed_datetime']
    valid_ends = df[end_candidates].copy()
    
    # Find the earliest valid end timestamp per row
    min_end_time = valid_ends.min(axis=1)
    
    # Calculate difference
    df['duration_minutes'] = (min_end_time - df['start_datetime']).dt.total_seconds() / 60.0
    
    # Any negative duration or extreme duration (e.g. over 30 days) is treated as outlier/error and set to NaN
    # 30 days in minutes = 30 * 24 * 60 = 43200 minutes
    df.loc[(df['duration_minutes'] < 0) | (df['duration_minutes'] > 43200), 'duration_minutes'] = np.nan
    
    # Impute missing durations using the median duration of similar events (event_cause + priority)
    median_durations = df.groupby(['event_cause', 'priority'])['duration_minutes'].transform('median')
    df['duration_minutes'] = df['duration_minutes'].fillna(median_durations)
    
    # If still missing (e.g. no events in that cause/priority group), use event_cause median
    cause_medians = df.groupby('event_cause')['duration_minutes'].transform('median')
    df['duration_minutes'] = df['duration_minutes'].fillna(cause_medians)
    
    # Finally, fill any remaining NaNs with global median duration
    global_median_duration = df['duration_minutes'].median()
    if pd.isna(global_median_duration):
        global_median_duration = 60.0  # Fallback default if whole column is NaN
    df['duration_minutes'] = df['duration_minutes'].fillna(global_median_duration)
    
    # 7. Remove duplicate records
    initial_rows = len(df)
    if 'id' in df.columns:
        # Drop duplicates based on the 'id' column
        df = df.drop_duplicates(subset=['id'], keep='first')
    else:
        # Drop duplicates based on all columns
        df = df.drop_duplicates(keep='first')
    
    duplicate_count = initial_rows - len(df)
    logger.info(f"Removed {duplicate_count} duplicate records. Remaining: {len(df)} rows.")
    
    return df

if __name__ == '__main__':
    # Simple verification test
    import sys
    test_file = r"c:\Users\Harshitha\Desktop\Grid lock project\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    if os.path.exists(test_file):
        processed_df = preprocess_data(test_file)
        print(processed_df[['id', 'event_type', 'event_cause', 'latitude', 'longitude', 'duration_minutes']].head())
    else:
        print("Test file not found, skipping verification.")
