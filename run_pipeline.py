import os
import sys
import pandas as pd
import logging
import json

# Add current directory to path so python can find the src modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.risk_engine import train_models, calculate_risk_score
from src.resource_recommender import recommend_resources_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Event-Driven Congestion Forecasting Pipeline...")
    
    # 1. Define paths
    raw_data_path = r"c:\Users\Harshitha\Desktop\Grid lock project\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
    processed_dir = r"c:\Users\Harshitha\Desktop\Grid lock project\data\processed"
    models_dir = r"c:\Users\Harshitha\Desktop\Grid lock project\models"
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Run Preprocessing
    try:
        processed_df = preprocess_data(raw_data_path)
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)
        
    # 3. Run Feature Engineering
    try:
        featured_df = engineer_features(processed_df, is_training=True)
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        sys.exit(1)
        
    # 4. Compute Risk Scores & Risk Levels
    logger.info("Computing risk scores and levels...")
    try:
        featured_df['risk_score'], featured_df['risk_level'] = calculate_risk_score(featured_df)
        logger.info(f"Risk levels distribution:\n{featured_df['risk_level'].value_counts()}")
    except Exception as e:
        logger.error(f"Risk scoring failed: {str(e)}")
        sys.exit(1)
        
    # 5. Train Machine Learning Models
    logger.info("Training predictive machine learning models...")
    try:
        metrics = train_models(featured_df, save_dir=models_dir)
        logger.info("Model training completed successfully.")
        logger.info(f"Classifier Accuracy: {metrics['classifier_accuracy']:.4f}")
        logger.info(f"Regressor MAE: {metrics['regressor_mae']:.2f} mins, R2 Score: {metrics['regressor_r2']:.4f}")
        
        # Save metrics to JSON file
        with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        sys.exit(1)
        
    # 6. Generate Resource Recommendations
    try:
        final_df = recommend_resources_df(featured_df)
    except Exception as e:
        logger.error(f"Resource recommendation failed: {str(e)}")
        sys.exit(1)
        
    # 7. Save Processed Outputs
    try:
        # Save complete featured dataset
        output_csv_path = os.path.join(processed_dir, 'processed_traffic_events.csv')
        final_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved complete processed dataset to {output_csv_path}")
        
        # Save simplified coordinates file for fast mapping/nearest-neighbor reference in Streamlit
        coords_path = os.path.join(processed_dir, 'historical_coords.csv')
        coords_df = final_df[['latitude', 'longitude', 'junction', 'zone']].drop_duplicates()
        coords_df.to_csv(coords_path, index=False)
        logger.info(f"Saved coordinate lookup file to {coords_path}")
        
    except Exception as e:
        logger.error(f"Saving processed outputs failed: {str(e)}")
        sys.exit(1)
        
    logger.info("Event-Driven Congestion Forecasting Pipeline executed successfully!")

if __name__ == '__main__':
    main()
