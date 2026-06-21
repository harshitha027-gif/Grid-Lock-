import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.neighbors import BallTree
import joblib
import logging

logger = logging.getLogger(__name__)

def calculate_risk_score(df: pd.DataFrame) -> tuple:
    """
    Apply a multi-criteria weighted scoring engine to compute a numeric risk score (0-100)
    and map it to categorical risk levels: Low, Medium, High, Critical.
    
    Parameters:
    df (pd.DataFrame): Dataframe with preprocessed and engineered features.
    
    Returns:
    tuple: (pd.Series of numeric scores, pd.Series of categorical labels)
    """
    df = df.copy()
    
    # 1. Road Closure Score (Weight: 25%)
    road_closure_score = df['requires_road_closure'].apply(lambda x: 100.0 if x else 0.0)
    
    # 2. Priority Score (Weight: 20%)
    priority_map = {'High': 100.0, 'Medium': 60.0, 'Low': 25.0}
    priority_score = df['priority'].map(priority_map).fillna(60.0)
    
    # 3. Cause Score (Weight: 20%)
    cause_scores = []
    for idx, row in df.iterrows():
        cause = str(row['event_cause']).lower().strip()
        event_type = str(row['event_type']).lower().strip()
        
        # Base cause severity
        if cause in ['accident', 'water_logging', 'congestion']:
            score = 100.0
        elif cause in ['public_event', 'construction', 'tree_fall']:
            score = 85.0
        elif cause in ['vehicle_breakdown']:
            score = 65.0
        elif cause in ['pot_holes', 'road_conditions']:
            score = 45.0
        else:
            score = 25.0
            
        # Add slight penalty for unplanned/unexpected events
        if event_type == 'unplanned':
            score = min(100.0, score + 5.0)
            
        cause_scores.append(score)
    cause_score = pd.Series(cause_scores, index=df.index)
    
    # 4. Duration Score (Weight: 20%)
    # Scale duration: 4 hours (240 mins) or more gets maximum duration score of 100
    duration_score = df['duration_minutes'].apply(lambda x: min(100.0, (x / 240.0) * 100.0) if not pd.isna(x) else 30.0)
    
    # 5. Density Score (Weight: 15%)
    # Scale using 95th percentile of historical density as the max reference
    p95_density = df['historical_event_density'].quantile(0.95)
    if pd.isna(p95_density) or p95_density == 0:
        p95_density = 10.0
    density_score = df['historical_event_density'].apply(lambda x: min(100.0, (x / p95_density) * 100.0))
    
    # Combine scores
    total_score = (
        0.25 * road_closure_score +
        0.20 * priority_score +
        0.20 * cause_score +
        0.20 * duration_score +
        0.15 * density_score
    )
    
    # Map to risk levels
    def map_score_to_level(score):
        if score < 35.0:
            return 'Low'
        elif score < 55.0:
            return 'Medium'
        elif score < 75.0:
            return 'High'
        else:
            return 'Critical'
            
    risk_level = total_score.apply(map_score_to_level)
    
    return total_score, risk_level

def train_models(df: pd.DataFrame, save_dir: str = 'models') -> dict:
    """
    Train and save ML models:
    1. Classifier for Risk Level (Low, Medium, High, Critical)
    2. Regressor for Duration (minutes)
    
    Parameters:
    df (pd.DataFrame): Featured dataset with risk labels calculated.
    save_dir (str): Directory to save trained model objects.
    
    Returns:
    dict: Model performance metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add risk labels to df for training
    df = df.copy()
    df['risk_score'], df['risk_level'] = calculate_risk_score(df)
    
    # Features & Targets
    feature_cols = [
        'event_type', 'event_cause', 'priority', 'requires_road_closure',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'zone_frequency', 'junction_frequency', 'historical_event_density'
    ]
    
    X = df[feature_cols]
    y_class = df['risk_level']
    y_reg = df['duration_minutes']
    
    # Identify feature types
    categorical_features = ['event_type', 'event_cause', 'priority']
    numeric_features = ['requires_road_closure', 'hour_of_day', 'day_of_week', 'is_weekend', 
                        'zone_frequency', 'junction_frequency', 'historical_event_density']
    
    # Build preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # ------------------- 1. Train Risk Classifier -------------------
    logger.info("Training Risk Classifier model...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    
    classifier_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    classifier_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = classifier_pipeline.predict(X_test_c)
    
    class_report = classification_report(y_test_c, y_pred_c, output_dict=True)
    logger.info(f"Classifier Accuracy: {class_report['accuracy']:.4f}")
    
    # Save classifier pipeline
    joblib.dump(classifier_pipeline, os.path.join(save_dir, 'risk_classifier.joblib'))
    
    # ------------------- 2. Train Duration Regressor -------------------
    logger.info("Training Duration Regressor model...")
    # Filter out records where duration is null or anomalous (though already cleaned in preprocessing)
    valid_duration_mask = y_reg.notna()
    X_reg = X[valid_duration_mask]
    y_reg_clean = y_reg[valid_duration_mask]
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg_clean, test_size=0.2, random_state=42)
    
    regressor_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    regressor_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = regressor_pipeline.predict(X_test_r)
    
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    logger.info(f"Regressor MAE: {mae:.2f} minutes, R2 Score: {r2:.4f}")
    
    # Save regressor pipeline
    joblib.dump(regressor_pipeline, os.path.join(save_dir, 'duration_regressor.joblib'))
    
    # Save 95th percentile density for reference during inference
    p95_density = df['historical_event_density'].quantile(0.95)
    joblib.dump(p95_density, os.path.join(save_dir, 'p95_density.joblib'))
    
    metrics = {
        'classifier_accuracy': class_report['accuracy'],
        'classifier_report': class_report,
        'regressor_mae': mae,
        'regressor_r2': r2
    }
    
    return metrics

def predict_event(model_dir: str, event_data: dict, historical_coords_df: pd.DataFrame = None) -> dict:
    """
    Predict risk level and forecast disruption duration for a single input event.
    
    Parameters:
    model_dir (str): Directory containing joblib models.
    event_data (dict): Dictionary with keys: event_type, event_cause, priority, 
                       requires_road_closure, start_datetime (string or Timestamp),
                       latitude, longitude, zone, junction.
    historical_coords_df (pd.DataFrame): DataFrame of coordinates for density calculation.
    
    Returns:
    dict: Prediction results including risk_score, risk_level, and predicted_duration.
    """
    # Load models
    classifier = joblib.load(os.path.join(model_dir, 'risk_classifier.joblib'))
    regressor = joblib.load(os.path.join(model_dir, 'duration_regressor.joblib'))
    p95_density = joblib.load(os.path.join(model_dir, 'p95_density.joblib'))
    
    # Build single row dataframe
    df = pd.DataFrame([event_data])
    
    # Convert types
    df['start_datetime'] = pd.to_datetime(df['start_datetime']).dt.tz_localize(None)
    df['requires_road_closure'] = df['requires_road_closure'].astype(bool)
    df['latitude'] = df['latitude'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    
    # Feature engineering for single row
    df['hour_of_day'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate density if historical data coordinates are supplied
    if historical_coords_df is not None and not historical_coords_df.empty:
        r_rad = 0.5 / 6371.0
        ref_coords = np.radians(historical_coords_df[['latitude', 'longitude']].values)
        query_coords = np.radians(df[['latitude', 'longitude']].values)
        tree = BallTree(ref_coords, metric='haversine')
        counts = tree.query_radius(query_coords, r=r_rad, count_only=True)
        df['historical_event_density'] = counts[0]
    else:
        df['historical_event_density'] = 1.0  # Fallback default
        
    # Set default values for frequency features if not supplied
    if 'zone_frequency' not in df.columns:
        df['zone_frequency'] = 100  # Reasonable average
    if 'junction_frequency' not in df.columns:
        df['junction_frequency'] = 10   # Reasonable average
        
    # Feature columns order alignment
    feature_cols = [
        'event_type', 'event_cause', 'priority', 'requires_road_closure',
        'hour_of_day', 'day_of_week', 'is_weekend',
        'zone_frequency', 'junction_frequency', 'historical_event_density'
    ]
    X = df[feature_cols]
    
    # Predict
    predicted_risk_level = classifier.predict(X)[0]
    predicted_duration = regressor.predict(X)[0]
    
    # Reconstruct the direct score formula for the simulation to give transparent details
    road_closure_score = 100.0 if event_data['requires_road_closure'] else 0.0
    priority_map = {'High': 100.0, 'Medium': 60.0, 'Low': 25.0}
    priority_score = priority_map.get(event_data['priority'], 60.0)
    
    cause = event_data['event_cause'].lower().strip()
    if cause in ['accident', 'water_logging', 'congestion']:
        cause_score = 100.0
    elif cause in ['public_event', 'construction', 'tree_fall']:
        cause_score = 85.0
    elif cause in ['vehicle_breakdown']:
        cause_score = 65.0
    elif cause in ['pot_holes', 'road_conditions']:
        cause_score = 45.0
    else:
        cause_score = 25.0
    if event_data['event_type'] == 'unplanned':
        cause_score = min(100.0, cause_score + 5.0)
        
    duration_score = min(100.0, (predicted_duration / 240.0) * 100.0)
    density_score = min(100.0, (df['historical_event_density'].iloc[0] / p95_density) * 100.0)
    
    risk_score = (
        0.25 * road_closure_score +
        0.20 * priority_score +
        0.20 * cause_score +
        0.20 * duration_score +
        0.15 * density_score
    )
    
    return {
        'risk_score': round(float(risk_score), 1),
        'risk_level': str(predicted_risk_level),
        'predicted_duration_minutes': round(float(predicted_duration), 1),
        'factor_breakdown': {
            'road_closure_score': round(float(road_closure_score), 1),
            'priority_score': round(float(priority_score), 1),
            'cause_score': round(float(cause_score), 1),
            'duration_score': round(float(duration_score), 1),
            'density_score': round(float(density_score), 1)
        }
    }
