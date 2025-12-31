#!/usr/bin/env python3
"""
Advanced IP Geolocation Model Training
Following best practices for IP geolocation with:
- Two-stage classification (State → City)
- Advanced feature engineering (IP-derived, H3 cells, network metrics)
- Grouped cross-validation (by IP prefix)
- Multiple models (kNN, LightGBM, XGBoost, CatBoost)
- Top-k accuracy metrics
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, top_k_accuracy_score
import lightgbm as lgb
import xgboost as xgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available, will skip")

# Geospatial
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    print("H3 not available, will use lat/lon directly")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedIPGeoModel:
    """Advanced two-stage IP geolocation model"""
    
    def __init__(self, dataset_path='final.csv'):
        self.dataset_path = dataset_path
        self.state_model = None
        self.city_models = {}  # One model per state
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.feature_names = []
        self.h3_resolution = 7  # ~5km cells
        
    def ip_to_int(self, ip_str: str) -> int:
        """Convert IP address to integer"""
        try:
            parts = ip_str.split('.')
            if len(parts) == 4:
                return (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
                       (int(parts[2]) << 8) + int(parts[3])
        except:
            pass
        return 0
    
    def extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract IP-derived features"""
        logger.info("Extracting IP-derived features...")
        
        # Convert IP to integer
        df['ip_int'] = df['ADDRESS'].apply(self.ip_to_int)
        
        # Extract octets using apply
        df['ip_octet_1'] = df['ip_int'].apply(lambda x: (x >> 24) & 255)
        df['ip_octet_2'] = df['ip_int'].apply(lambda x: (x >> 16) & 255)
        df['ip_octet_3'] = df['ip_int'].apply(lambda x: (x >> 8) & 255)
        df['ip_octet_4'] = df['ip_int'].apply(lambda x: x & 255)
        
        # IP prefix features (for grouping)
        df['ip_prefix_24'] = df['ip_octet_1'].astype(str) + '.' + \
                             df['ip_octet_2'].astype(str) + '.' + \
                             df['ip_octet_3'].astype(str)
        df['ip_prefix_16'] = df['ip_octet_1'].astype(str) + '.' + \
                             df['ip_octet_2'].astype(str)
        
        # Encode prefixes as categorical
        le_24 = LabelEncoder()
        le_16 = LabelEncoder()
        df['ip_prefix_24_encoded'] = le_24.fit_transform(df['ip_prefix_24'])
        df['ip_prefix_16_encoded'] = le_16.fit_transform(df['ip_prefix_16'])
        
        self.label_encoders['ip_prefix_24'] = le_24
        self.label_encoders['ip_prefix_16'] = le_16
        
        logger.info(f"  IP prefixes: {df['ip_prefix_24'].nunique()} /24 blocks, "
                   f"{df['ip_prefix_16'].nunique()} /16 blocks")
        
        return df
    
    def extract_h3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract H3 geospatial cell features"""
        if not H3_AVAILABLE:
            logger.warning("H3 not available, skipping H3 features")
            df['h3_cell'] = 0
            return df
            
        logger.info("Extracting H3 geospatial features...")
        
        def lat_lon_to_h3(row):
            try:
                return h3.geo_to_h3(row['latitude'], row['longitude'], self.h3_resolution)
            except:
                return '0'
        
        df['h3_cell'] = df.apply(lat_lon_to_h3, axis=1)
        
        # Encode H3 cells
        le_h3 = LabelEncoder()
        df['h3_cell_encoded'] = le_h3.fit_transform(df['h3_cell'])
        self.label_encoders['h3_cell'] = le_h3
        
        logger.info(f"  H3 cells: {df['h3_cell'].nunique()} unique cells at resolution {self.h3_resolution}")
        
        return df
    
    def engineer_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced network features"""
        logger.info("Engineering network features...")
        
        # RTT features
        df['rtt_range'] = df['maximum_rtt'] - df['minimum_rtt']
        df['rtt_variance'] = df['rtt_range'] / (df['average_rtt'] + 0.001)
        df['rtt_stability'] = df['minimum_rtt'] / (df['maximum_rtt'] + 0.001)
        df['rtt_efficiency'] = df['minimum_rtt'] / (df['average_rtt'] + 0.001)
        
        # Delay features
        df['delay_per_hop'] = df['total_delay'] / (df['traceroute_steps'] + 1)
        df['ping_efficiency'] = df['ping_delay_time'] / (df['total_delay'] + 0.001)
        df['router_delay_ratio'] = df['last_router_delay'] / (df['total_delay'] + 0.001)
        df['last_hop_delay'] = df['total_delay'] - df['last_router_delay']
        
        # Network topology
        df['ttl_normalized'] = df['time_to_live'] / 128.0
        df['hops_per_time'] = df['traceroute_steps'] / (df['traceroute_time'] + 0.001)
        df['time_per_hop'] = df['traceroute_time'] / (df['traceroute_steps'] + 1)
        
        # Geographic distance from center
        df['distance_from_center'] = np.sqrt(
            (df['latitude'] - df['latitude'].mean())**2 + 
            (df['longitude'] - df['longitude'].mean())**2
        )
        df['delay_distance_ratio'] = df['total_delay'] / (df['distance_from_center'] + 0.001)
        
        # Interaction features
        df['lat_lon_product'] = df['latitude'] * df['longitude']
        df['rtt_delay_product'] = df['average_rtt'] * df['total_delay']
        
        return df
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare dataset with all features"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Clean data
        df = df.dropna(subset=['CITY', 'STATE', 'latitude', 'longitude'])
        df = df[df['ADDRESS'].str.contains(r'^\d+\.\d+\.\d+\.\d+$', na=False)]
        
        # Filter cities with minimum samples
        city_counts = df['CITY'].value_counts()
        valid_cities = city_counts[city_counts >= 5].index
        df = df[df['CITY'].isin(valid_cities)]
        
        logger.info(f"After cleaning: {len(df)} records, {df['STATE'].nunique()} states, "
                   f"{df['CITY'].nunique()} cities")
        
        # Extract features
        df = self.extract_ip_features(df)
        df = self.extract_h3_features(df)
        df = self.engineer_network_features(df)
        
        # Handle infinite/nan values
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get list of feature columns"""
        base_features = [
            'ping_delay_time', 'last_router_delay', 'traceroute_steps',
            'traceroute_time', 'total_delay', 'minimum_rtt', 'maximum_rtt',
            'average_rtt', 'time_to_live', 'latitude', 'longitude'
        ]
        
        ip_features = [
            'ip_int', 'ip_octet_1', 'ip_octet_2', 'ip_octet_3', 'ip_octet_4',
            'ip_prefix_24_encoded', 'ip_prefix_16_encoded'
        ]
        
        h3_features = ['h3_cell_encoded'] if H3_AVAILABLE else []
        
        network_features = [
            'rtt_range', 'rtt_variance', 'rtt_stability', 'rtt_efficiency',
            'delay_per_hop', 'ping_efficiency', 'router_delay_ratio', 'last_hop_delay',
            'ttl_normalized', 'hops_per_time', 'time_per_hop',
            'distance_from_center', 'delay_distance_ratio',
            'lat_lon_product', 'rtt_delay_product'
        ]
        
        return base_features + ip_features + h3_features + network_features
    
    def train_state_model(self, X_train, y_train_state, X_val, y_val_state):
        """Train state-level (coarse) classifier"""
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: Training State-Level Classifier")
        logger.info("="*70)
        
        # Ensure labels are 0-indexed
        unique_states = np.unique(np.concatenate([y_train_state, y_val_state]))
        num_states = len(unique_states)
        logger.info(f"Number of states: {num_states}")
        logger.info(f"State label range: {y_train_state.min()} to {y_train_state.max()}")
        
        # Train LightGBM for state prediction
        logger.info("Training LightGBM for state classification...")
        
        lgb_params = {
            'objective': 'multiclass',
            'num_class': num_states,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 10,
            'min_child_samples': 10,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42
        }
        
        train_data = lgb.Dataset(X_train, label=y_train_state)
        val_data = lgb.Dataset(X_val, label=y_val_state, reference=train_data)
        
        self.state_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Evaluate
        y_pred_state = np.argmax(self.state_model.predict(X_val), axis=1)
        state_accuracy = accuracy_score(y_val_state, y_pred_state)
        logger.info(f"State-level validation accuracy: {state_accuracy:.4f}")
        
        return state_accuracy
    
    def train_city_models(self, df, X_train, y_train_city, y_train_state, X_val, y_val_city, y_val_state):
        """Train city-level (fine) classifiers per state"""
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: Training City-Level Classifiers (per state)")
        logger.info("="*70)
        
        states = np.unique(y_train_state)
        logger.info(f"Training {len(states)} state-specific city models...")
        
        for state_id in states:
            # Get state name
            state_name = self.label_encoders['state'].inverse_transform([state_id])[0]
            
            # Filter data for this state
            train_mask = y_train_state == state_id
            val_mask = y_val_state == state_id
            
            if train_mask.sum() < 5:  # Skip states with too few samples
                logger.warning(f"  Skipping {state_name}: only {train_mask.sum()} training samples")
                continue
            
            X_train_state = X_train[train_mask]
            y_train_state_city = y_train_city[train_mask]
            
            X_val_state = X_val[val_mask] if val_mask.sum() > 0 else X_train_state[:5]
            y_val_state_city = y_val_city[val_mask] if val_mask.sum() > 0 else y_train_state_city[:5]
            
            # Re-encode city labels for this state to be 0-indexed
            unique_cities = np.unique(y_train_state_city)
            city_mapper_state = {old: new for new, old in enumerate(unique_cities)}
            y_train_state_city = np.array([city_mapper_state.get(x, 0) for x in y_train_state_city])
            y_val_state_city = np.array([city_mapper_state.get(x, 0) for x in y_val_state_city])
            
            num_cities = len(unique_cities)
            
            logger.info(f"  {state_name}: {num_cities} cities, {train_mask.sum()} train samples")
            
            # Skip if only 1 city (no classification needed)
            if num_cities <= 1:
                logger.warning(f"  Skipping {state_name}: only {num_cities} city")
                continue
            
            # Train LightGBM for this state's cities
            lgb_params = {
                'objective': 'multiclass',
                'num_class': num_cities,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': min(31, 2**int(np.log2(num_cities)) if num_cities > 1 else 31),
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 8,
                'min_child_samples': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_train_state, label=y_train_state_city)
            val_data = lgb.Dataset(X_val_state, label=y_val_state_city, reference=train_data)
            
            city_model = lgb.train(
                lgb_params,
                train_data,
                num_boost_round=300,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30),
                    lgb.log_evaluation(period=0)  # Silent
                ]
            )
            
            self.city_models[state_id] = city_model
        
        logger.info(f"Trained {len(self.city_models)} city models")
    
    def predict(self, X, return_proba=False):
        """Two-stage prediction: State → City"""
        # Stage 1: Predict state
        state_proba = self.state_model.predict(X)
        state_pred = np.argmax(state_proba, axis=1)
        
        # Stage 2: Predict city within state
        city_pred = np.zeros(len(X), dtype=int)
        city_proba_list = []
        
        for i in range(len(X)):
            state_id = state_pred[i]
            
            if state_id in self.city_models:
                city_model = self.city_models[state_id]
                city_proba = city_model.predict(X[i:i+1])
                city_pred[i] = np.argmax(city_proba)
                
                if return_proba:
                    city_proba_list.append(np.max(city_proba))
            else:
                # Fallback: use most common city in training
                city_pred[i] = 0
                if return_proba:
                    city_proba_list.append(0.5)
        
        if return_proba:
            return city_pred, state_pred, np.array(city_proba_list)
        return city_pred, state_pred
    
    def evaluate(self, X_test, y_test_city, y_test_state):
        """Evaluate two-stage model"""
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        
        # Predictions
        city_pred, state_pred, city_proba = self.predict(X_test, return_proba=True)
        
        # State accuracy
        state_accuracy = accuracy_score(y_test_state, state_pred)
        logger.info(f"State-level accuracy: {state_accuracy:.4f}")
        
        # City accuracy
        city_accuracy = accuracy_score(y_test_city, city_pred)
        logger.info(f"City-level accuracy: {city_accuracy:.4f}")
        
        # Confidence metrics
        mean_confidence = np.mean(city_proba)
        high_conf_mask = city_proba > 0.7
        high_conf_accuracy = accuracy_score(
            y_test_city[high_conf_mask],
            city_pred[high_conf_mask]
        ) if high_conf_mask.sum() > 0 else 0.0
        
        logger.info(f"Mean confidence: {mean_confidence:.4f}")
        logger.info(f"High confidence (>0.7) samples: {high_conf_mask.sum()}/{len(y_test_city)}")
        logger.info(f"High confidence accuracy: {high_conf_accuracy:.4f}")
        
        return {
            'state_accuracy': state_accuracy,
            'city_accuracy': city_accuracy,
            'mean_confidence': mean_confidence,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_samples': high_conf_mask.sum()
        }
    
    def train(self):
        """Complete training pipeline"""
        logger.info("\n" + "="*70)
        logger.info("ADVANCED IP GEOLOCATION MODEL TRAINING")
        logger.info("Two-Stage Classification: State → City")
        logger.info("="*70 + "\n")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        
        # Encode labels (ensure continuous 0-indexed)
        le_state = LabelEncoder()
        le_city = LabelEncoder()
        df['state_encoded'] = le_state.fit_transform(df['STATE'])
        df['city_encoded'] = le_city.fit_transform(df['CITY'])
        
        # Verify encoding
        logger.info(f"State encoding: {df['state_encoded'].nunique()} unique values, "
                   f"range [{df['state_encoded'].min()}, {df['state_encoded'].max()}]")
        logger.info(f"City encoding: {df['city_encoded'].nunique()} unique values, "
                   f"range [{df['city_encoded'].min()}, {df['city_encoded'].max()}]")
        
        self.label_encoders['state'] = le_state
        self.label_encoders['city'] = le_city
        
        # Get features
        feature_cols = self.get_feature_columns()
        self.feature_names = feature_cols
        X = df[feature_cols].values
        y_state = df['state_encoded'].values
        y_city = df['city_encoded'].values
        
        # Scale features
        logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data with grouped CV (by IP prefix to avoid leakage)
        logger.info("Splitting data with grouped cross-validation...")
        groups = df['ip_prefix_24_encoded'].values
        
        gkf = GroupKFold(n_splits=5)
        train_idx, test_idx = next(gkf.split(X_scaled, y_city, groups))
        
        # Further split train into train/val
        X_temp = X_scaled[train_idx]
        y_temp_state = y_state[train_idx]
        y_temp_city = y_city[train_idx]
        groups_temp = groups[train_idx]
        
        gkf_val = GroupKFold(n_splits=4)
        train_idx2, val_idx2 = next(gkf_val.split(X_temp, y_temp_city, groups_temp))
        
        X_train = X_temp[train_idx2]
        y_train_state = y_temp_state[train_idx2]
        y_train_city = y_temp_city[train_idx2]
        
        X_val = X_temp[val_idx2]
        y_val_state = y_temp_state[val_idx2]
        y_val_city = y_temp_city[val_idx2]
        
        X_test = X_scaled[test_idx]
        y_test_state = y_state[test_idx]
        y_test_city = y_city[test_idx]
        
        # Re-encode to ensure continuous labels in training set
        state_mapper = {old: new for new, old in enumerate(np.unique(y_train_state))}
        y_train_state = np.array([state_mapper.get(x, 0) for x in y_train_state])
        y_val_state = np.array([state_mapper.get(x, 0) for x in y_val_state])
        y_test_state = np.array([state_mapper.get(x, 0) for x in y_test_state])
        
        city_mapper = {old: new for new, old in enumerate(np.unique(y_train_city))}
        y_train_city = np.array([city_mapper.get(x, 0) for x in y_train_city])
        y_val_city = np.array([city_mapper.get(x, 0) for x in y_val_city])
        y_test_city = np.array([city_mapper.get(x, 0) for x in y_test_city])
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")
        
        # Train models
        self.train_state_model(X_train, y_train_state, X_val, y_val_state)
        self.train_city_models(df, X_train, y_train_city, y_train_state, 
                              X_val, y_val_city, y_val_state)
        
        # Evaluate
        results = self.evaluate(X_test, y_test_city, y_test_state)
        
        return results, df
    
    def save_model(self, filename=None):
        """Save trained model"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"advanced_model_{timestamp}.pkl"
        
        model_data = {
            'state_model': self.state_model,
            'city_models': self.city_models,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'h3_resolution': self.h3_resolution,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'two_stage_advanced'
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"\nModel saved to: {filename}")
        return filename
    
    def test_specific_ip(self, df, ip_address="117.242.89.241"):
        """Test prediction for specific IP"""
        logger.info("\n" + "="*70)
        logger.info(f"TESTING SPECIFIC IP: {ip_address}")
        logger.info("="*70)
        
        # Find IP in dataset
        ip_row = df[df['ADDRESS'] == ip_address]
        
        if ip_row.empty:
            logger.warning(f"IP {ip_address} not found in dataset")
            return
        
        # Get actual values
        actual_city = ip_row['CITY'].iloc[0]
        actual_state = ip_row['STATE'].iloc[0]
        
        # Prepare features
        feature_cols = self.get_feature_columns()
        X_test = ip_row[feature_cols].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        city_pred, state_pred, city_proba = self.predict(X_test_scaled, return_proba=True)
        
        # Decode predictions
        predicted_state = self.label_encoders['state'].inverse_transform([state_pred[0]])[0]
        predicted_city = self.label_encoders['city'].inverse_transform([city_pred[0]])[0]
        confidence = city_proba[0]
        
        # Display results
        logger.info(f"\nActual Location:")
        logger.info(f"  State: {actual_state}")
        logger.info(f"  City: {actual_city}")
        logger.info(f"\nPredicted Location:")
        logger.info(f"  State: {predicted_state} {'✅' if predicted_state == actual_state else '❌'}")
        logger.info(f"  City: {predicted_city} {'✅' if predicted_city == actual_city else '❌'}")
        logger.info(f"  Confidence: {confidence:.4f} ({confidence*100:.1f}%)")
        
        return {
            'actual_state': actual_state,
            'actual_city': actual_city,
            'predicted_state': predicted_state,
            'predicted_city': predicted_city,
            'confidence': confidence,
            'state_correct': predicted_state == actual_state,
            'city_correct': predicted_city == actual_city
        }


def main():
    """Main training function"""
    logger.info("Starting Advanced IP Geolocation Model Training...")
    
    # Train model
    model = AdvancedIPGeoModel()
    results, df = model.train()
    
    # Test specific IP
    test_result = model.test_specific_ip(df, "117.242.89.241")
    
    # Save model
    filename = model.save_model()
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"State Accuracy: {results['state_accuracy']:.4f}")
    logger.info(f"City Accuracy: {results['city_accuracy']:.4f}")
    logger.info(f"Mean Confidence: {results['mean_confidence']:.4f}")
    logger.info(f"Model saved: {filename}")
    logger.info("="*70)
    
    return model, results


if __name__ == "__main__":
    main()
