#!/usr/bin/env python3
"""
Fast XGBoost Regression Training for IP Geolocation
Predicts Latitude/Longitude directly from network metrics.
Maps predictions to nearest city using Haversine distance.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from math import radians, cos, sin, asin, sqrt
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

class GeoIPRegressor:
    def __init__(self):
        self.model = None
        self.city_centroids = {}
        self.feature_names = []

    def engineer_features(self, df):
        """Create derived network features"""
        logger.info("Engineering features...")
        
        # Ensure required columns exist
        required_cols = ['maximum_rtt', 'minimum_rtt', 'average_rtt', 
                        'total_delay', 'traceroute_steps', 'traceroute_time', 'time_to_live']
        
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Column {col} missing, filling with defaults")
                df[col] = 0
                
        # Basic RTT stats
        df['rtt_range'] = df['maximum_rtt'] - df['minimum_rtt']
        df['rtt_variance'] = (df['maximum_rtt'] - df['average_rtt'])**2
        
        # Hop based metrics
        # Avoid division by zero
        df['delay_per_hop'] = df['total_delay'] / (df['traceroute_steps'] + 1)
        df['hop_latency'] = df['traceroute_time'] / (df['traceroute_steps'] + 1)
        
        # Ratios
        df['min_max_ratio'] = df['minimum_rtt'] / (df['maximum_rtt'] + 0.1)
        df['avg_total_ratio'] = df['average_rtt'] / (df['total_delay'] + 0.1)
        
        # TTL bucketing (categoricalish)
        df['ttl_bucket'] = (df['time_to_live'] // 10) * 10
        
        return df

    def prepare_data(self, filepath='ml_data_pipeline/final_cleaned.csv'):
        """Load and clean data"""
        logger.info(f"Loading data from {filepath}")
        
        # Handle different paths for execution
        if not os.path.exists(filepath):
            # Try src relative path
            alt_path = os.path.join(os.path.dirname(__file__), 'ml_data_pipeline/final_cleaned.csv')
            if os.path.exists(alt_path):
                filepath = alt_path
            else:
                 # Try absolute fallback
                 filepath = '/home/sanlatpi/HexaSentinel-main/src/ml_data_pipeline/final_cleaned.csv'
        
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
        logger.info(f"Loaded {len(df)} rows")
        
        # Filter valid lat/lon
        df = df.dropna(subset=['latitude', 'longitude', 'CITY'])
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Define features to use
        self.feature_names = [
            'ping_delay_time', 'last_router_delay', 'traceroute_steps', 
            'traceroute_time', 'total_delay', 'minimum_rtt', 'maximum_rtt', 
            'average_rtt', 'time_to_live',
            'rtt_range', 'rtt_variance', 'delay_per_hop', 'hop_latency',
            'min_max_ratio', 'avg_total_ratio', 'ttl_bucket'
        ]
        
        # Ensure all feature columns exist, fill NaNs
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0)
            
        X = df[self.feature_names]
        # Multi-output target: Lat, Lon
        y = df[['latitude', 'longitude']]
        
        # Keep metadata for evaluation
        meta = df[['CITY', 'STATE', 'latitude', 'longitude']]
        
        # Build city centroids dictionary for mapping later
        # Group by City and take mean of lat/lon
        centroids = df.groupby(['CITY', 'STATE'])[['latitude', 'longitude']].mean()
        
        # Store as dictionary: "City, State" -> {'latitude': x, 'longitude': y}
        self.city_centroids = {}
        for idx, row in centroids.iterrows():
            city, state = idx
            key = f"{city}||{state}" # Unique key
            self.city_centroids[key] = {
                'city': city,
                'state': state,
                'latitude': row['latitude'],
                'longitude': row['longitude']
            }
            
        logger.info(f"Built centroids for {len(self.city_centroids)} cities")
        
        return X, y, meta

    def train(self, X_train, y_train):
        """Train XGBoost Regressor"""
        logger.info("Training XGBoost Regressor...")
        
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")

    def find_nearest_city(self, lat, lon):
        """Find nearest city from centroids using Haversine"""
        min_dist = float('inf')
        nearest_city_data = None
        
        for key, data in self.city_centroids.items():
            dist = haversine(lon, lat, data['longitude'], data['latitude'])
            if dist < min_dist:
                min_dist = dist
                nearest_city_data = data
                
        return nearest_city_data, min_dist

    def find_top_k_cities(self, lat, lon, k=3):
        """Return top k nearest cities"""
        distances = []
        for key, data in self.city_centroids.items():
            dist = haversine(lon, lat, data['longitude'], data['latitude'])
            distances.append((data, dist))
            
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def evaluate(self, X_test, y_test, meta_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        preds = self.model.predict(X_test)
        
        # Distance errors
        distances = []
        accuracies_top1 = []
        accuracies_top3 = []
        
        # Convert y_test to numpy for easier indexing if it's a dataframe
        y_test_np = y_test.values
        meta_test_reset = meta_test.reset_index(drop=True)
        
        for i in range(len(preds)):
            pred_lat, pred_lon = preds[i]
            true_lat, true_lon = y_test_np[i]
            
            # Calculate physical distance error
            dist_error = haversine(pred_lon, pred_lat, true_lon, true_lat)
            distances.append(dist_error)
            
            # City Mapping Check
            actual_city = meta_test_reset.iloc[i]['CITY']
            actual_state = meta_test_reset.iloc[i]['STATE']
            
            # Top 1
            nearest_data, _ = self.find_nearest_city(pred_lat, pred_lon)
            if nearest_data:
                match = (nearest_data['city'] == actual_city and nearest_data['state'] == actual_state)
                accuracies_top1.append(1 if match else 0)
            else:
                accuracies_top1.append(0)
            
            # Top 3
            top_k = self.find_top_k_cities(pred_lat, pred_lon, k=3)
            # Check if actual is in top k
            hit = False
            for item in top_k:
                data = item[0]
                if data['city'] == actual_city and data['state'] == actual_state:
                    hit = True
                    break
            accuracies_top3.append(1 if hit else 0)
            
        median_error = np.median(distances)
        mean_error = np.mean(distances)
        acc_1 = np.mean(accuracies_top1)
        acc_3 = np.mean(accuracies_top3)
        
        logger.info(f"="*40)
        logger.info(f"RESULTS")
        logger.info(f"="*40)
        logger.info(f"Median Distance Error: {median_error:.2f} km")
        logger.info(f"Mean Distance Error:   {mean_error:.2f} km")
        logger.info(f"City Accuracy (Top-1): {acc_1 * 100:.2f}%")
        logger.info(f"City Accuracy (Top-3): {acc_3 * 100:.2f}%")
        logger.info(f"="*40)
        
        return median_error, acc_1

    def save(self, path='advanced_loc_model.joblib'):
        """Save model and artifacts"""
        logger.info(f"Saving model to {path}")
        package = {
            'model': self.model,
            'city_centroids': self.city_centroids,
            'feature_names': self.feature_names,
            'model_type': 'xgboost_regression_haversine'
        }
        joblib.dump(package, path)
        logger.info("Model saved successfully")

if __name__ == "__main__":
    # Initialize
    geo_reg = GeoIPRegressor()
    
    # Run pipeline
    data_path = 'ml_data_pipeline/final_cleaned.csv' # Default relative path
    X, y, meta = geo_reg.prepare_data(data_path)
    
    # Split
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )
    
    # Train
    geo_reg.train(X_train, y_train)
    
    # Evaluate
    geo_reg.evaluate(X_test, y_test, meta_test)
    
    # Save
    geo_reg.save('advanced_loc_model.joblib')
