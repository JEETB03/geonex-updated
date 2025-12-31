#!/usr/bin/env python3
"""
Integrated Model Loader
Combines the advanced ML model (pkl) with the GeoModel (joblib)
Provides a unified interface for IP geolocation predictions
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import glob
import os
from math import radians, cos, sin, asin, sqrt

class IntegratedIPGeolocator:
    """
    Integrated IP Geolocation System
    Combines:
    1. Advanced two-stage ML model (State → City) from pkl or joblib
    2. GeoModel with API fallback and caching from joblib or pkl
    
    Supports both .pkl and .joblib model formats for flexibility
    """
    
    def __init__(self):
        self.advanced_model = None
        self.geo_model = None
        self.dataset = None
    
    def load_model_file(self, filepath: str):
        """
        Load a model from file - automatically detects .pkl or .joblib format
        
        Args:
            filepath: Path to model file (.pkl or .joblib)
            
        Returns:
            Loaded model object
        """
        if filepath.endswith('.joblib'):
            return joblib.load(filepath)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            # Try both methods
            try:
                return joblib.load(filepath)
            except:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
        
    def load_models(self) -> bool:
        """Load both models - supports .pkl and .joblib formats"""
        success = True
        
        # Load advanced model (try explicit regression model first, then standard classification)
        try:
            # Try regression model first
            if os.path.exists("advanced_loc_model.joblib"):
                self.advanced_model = joblib.load("advanced_loc_model.joblib")
                print(f"✅ Advanced model loaded: advanced_loc_model.joblib (Regression)")
            else:
                # Try .pkl files first
                model_files = glob.glob("advanced_model_*.pkl")
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    with open(latest_model, 'rb') as f:
                        self.advanced_model = pickle.load(f)
                    print(f"✅ Advanced model loaded: {latest_model} (Classification)")
                else:
                    # Try .joblib files as fallback
                    model_files = glob.glob("advanced_model_*.joblib")
                    if model_files:
                        latest_model = max(model_files, key=os.path.getctime)
                        self.advanced_model = joblib.load(latest_model)
                        print(f"✅ Advanced model loaded: {latest_model} (Classification)")
                    else:
                        print("⚠️  No advanced model found")
                        success = False
        except Exception as e:
            print(f"❌ Failed to load advanced model: {e}")
            success = False
        
        # Load GeoModel (joblib or pkl)
        try:
            # Try .joblib first
            if os.path.exists('model.joblib'):
                self.geo_model = joblib.load('model.joblib')
                print("✅ GeoModel loaded: model.joblib")
            elif os.path.exists('geomodel.joblib'):
                self.geo_model = joblib.load('geomodel.joblib')
                print("✅ GeoModel loaded: geomodel.joblib")
            # Try .pkl as fallback
            elif os.path.exists('model.pkl'):
                with open('model.pkl', 'rb') as f:
                    self.geo_model = pickle.load(f)
                print("✅ GeoModel loaded: model.pkl")
            elif os.path.exists('geomodel.pkl'):
                with open('geomodel.pkl', 'rb') as f:
                    self.geo_model = pickle.load(f)
                print("✅ GeoModel loaded: geomodel.pkl")
            else:
                print("⚠️  No GeoModel found")
        except Exception as e:
            print(f"❌ Failed to load GeoModel: {e}")
        
        # Load dataset
        try:
            if os.path.exists('final.csv'):
                self.dataset = pd.read_csv('final.csv')
                print(f"✅ Dataset loaded: {len(self.dataset)} records")
            else:
                print("⚠️  No dataset found (final.csv)")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
        
        return success
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers
        return c * r

    def find_nearest_city_regression(self, lat, lon):
        """Find nearest city from centroids using Haversine (for regression model)"""
        if not self.advanced_model or 'city_centroids' not in self.advanced_model:
            return "Unknown", "Unknown", 0.0
            
        city_centroids = self.advanced_model['city_centroids']
        min_dist = float('inf')
        nearest_data = None
        
        for key, data in city_centroids.items():
            dist = self.haversine(lon, lat, data['longitude'], data['latitude'])
            if dist < min_dist:
                min_dist = dist
                nearest_data = data
                
        if nearest_data:
            # Simple confidence estimation based on distance
            # Closer = Higher confidence
            # 0km = 1.0, 100km = 0.8, 500km+ = 0.5
            confidence = max(0.5, 1.0 - (min_dist / 500.0))
            return nearest_data['state'], nearest_data['city'], confidence
            
        return "Unknown", "Unknown", 0.0

    def predict_regression(self, ip_address: str, network_metrics: Dict) -> Optional[Dict]:
        """
        Predict using XGBoost Regression model (Lat/Lon)
        """
        if not self.advanced_model or self.advanced_model.get('model_type') != 'xgboost_regression_haversine':
            return None
            
        try:
            # Engineer features (must match training script)
            # Basic metrics with defaults
            max_rtt = network_metrics.get('maximum_rtt', 0)
            min_rtt = network_metrics.get('minimum_rtt', 0)
            avg_rtt = network_metrics.get('average_rtt', 0)
            total_delay = network_metrics.get('total_delay', 0)
            steps = network_metrics.get('traceroute_steps', 0)
            trace_time = network_metrics.get('traceroute_time', 0)
            ttl = network_metrics.get('time_to_live', 0)
            
            ping_delay = network_metrics.get('ping_delay_time', 0)
            last_router = network_metrics.get('last_router_delay', 0)
            
            # Derived features
            rtt_range = max_rtt - min_rtt
            rtt_variance = (max_rtt - avg_rtt)**2
            delay_per_hop = total_delay / (steps + 1) if steps >= 0 else 0
            hop_latency = trace_time / (steps + 1) if steps >= 0 else 0
            min_max_ratio = min_rtt / (max_rtt + 0.1)
            avg_total_ratio = avg_rtt / (total_delay + 0.1)
            ttl_bucket = (ttl // 10) * 10
            
            # Construct feature vector
            feature_names = self.advanced_model.get('feature_names', [])
            
            # Create a dictionary first locally
            features_dict = {
                'ping_delay_time': ping_delay,
                'last_router_delay': last_router,
                'traceroute_steps': steps,
                'traceroute_time': trace_time,
                'total_delay': total_delay,
                'minimum_rtt': min_rtt,
                'maximum_rtt': max_rtt,
                'average_rtt': avg_rtt,
                'time_to_live': ttl,
                'rtt_range': rtt_range,
                'rtt_variance': rtt_variance,
                'delay_per_hop': delay_per_hop,
                'hop_latency': hop_latency,
                'min_max_ratio': min_max_ratio,
                'avg_total_ratio': avg_total_ratio,
                'ttl_bucket': ttl_bucket
            }
            
            # Ensure order matches feature_names
            input_vector = []
            for name in feature_names:
                input_vector.append(features_dict.get(name, 0))
                
            input_df = pd.DataFrame([input_vector], columns=feature_names)
            
            # Predict
            model = self.advanced_model['model']
            pred = model.predict(input_df)[0]
            pred_lat, pred_lon = float(pred[0]), float(pred[1])
            
            # Find nearest city
            state, city, confidence = self.find_nearest_city_regression(pred_lat, pred_lon)
            
            return {
                'ip': ip_address,
                'country': 'India',
                'state': state,
                'city': city,
                'lat': pred_lat,
                'lon': pred_lon,
                'confidence': confidence,
                'method': 'XGBoost Regression',
                'model_type': 'regression_haversine'
            }
            
        except Exception as e:
            print(f"Regression prediction error: {e}")
            return None

    def predict_advanced(self, ip_address: str, network_metrics: Dict) -> Optional[Dict]:
        """
        Predict using legacy advanced two-stage model (State → City)
        """
        if not self.advanced_model:
            return None
            
        # If it's the new regression model, use the regression predictor
        if isinstance(self.advanced_model, dict) and self.advanced_model.get('model_type') == 'xgboost_regression_haversine':
            return self.predict_regression(ip_address, network_metrics)
            
        # Legacy Classification Logic...
        try:
             # Engineer all 34 features (Legacy)
            ping_delay = network_metrics.get('ping_delay_time', 60.0)
            
            # ... (Rest of legacy extraction - Stubbed for now as we focus on regression)
            pass
            
        except Exception as e:
            print(f"Advanced model prediction error: {e}")
            return None
        return None

    def predict_geomodel(self, ip_address: str) -> Optional[Dict]:
        """
        Predict using GeoModel (with API fallback and caching)
        Works without network metrics
        """
        if not self.geo_model:
            return None
        
        try:
            state, city, lat, lon = self.geo_model.predict(ip_address)
            
            return {
                'ip': ip_address,
                'country': 'India',
                'state': state,
                'city': city,
                'lat': lat,
                'lon': lon,
                'confidence': 0.85,  # GeoModel uses API, so high confidence
                'method': 'GeoModel (API + Cache)',
                'model_type': 'hybrid'
            }
            
        except Exception as e:
            print(f"GeoModel prediction error: {e}")
            return None
    
    def predict_hybrid(self, ip_address: str, network_metrics: Optional[Dict] = None) -> Dict:
        """
        Hybrid prediction: Try advanced model first, fallback to GeoModel
        """
        result = {
            'ip': ip_address,
            'predictions': [],
            'primary': None,
            'fallback': None
        }
        
        # Try advanced model if network metrics available
        if network_metrics and self.advanced_model:
            # Check model type to call correct method
            if isinstance(self.advanced_model, dict) and self.advanced_model.get('model_type') == 'xgboost_regression_haversine':
                advanced_pred = self.predict_regression(ip_address, network_metrics)
            else:
                 # Legacy
                advanced_pred = self.predict_advanced(ip_address, network_metrics)
                
            if advanced_pred:
                result['predictions'].append(advanced_pred)
                result['primary'] = advanced_pred
        
        # Try GeoModel
        if self.geo_model:
            geo_pred = self.predict_geomodel(ip_address)
            if geo_pred:
                result['predictions'].append(geo_pred)
                if not result['primary']:
                    result['primary'] = geo_pred
                else:
                    result['fallback'] = geo_pred
        
        # If no predictions, return error
        if not result['predictions']:
            result['primary'] = {
                'ip': ip_address,
                'country': 'Unknown',
                'state': 'Unknown',
                'city': 'Unknown',
                'lat': 0.0,
                'lon': 0.0,
                'confidence': 0.0,
                'method': 'No models available',
                'model_type': 'none'
            }
        
        return result
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        is_regression = isinstance(self.advanced_model, dict) and self.advanced_model.get('model_type') == 'xgboost_regression_haversine'
        
        info = {
            'advanced_model': {
                'loaded': self.advanced_model is not None,
                'type': self.advanced_model.get('model_type') if is_regression else 'classification (legacy)',
                'features': len(self.advanced_model.get('feature_names', [])) if is_regression else 0
            },
            'geo_model': {
                'loaded': self.geo_model is not None,
                'type': 'GeoModel' if self.geo_model else None,
                'has_cache': hasattr(self.geo_model, 'cache') if self.geo_model else False,
                'cache_size': len(self.geo_model.cache) if self.geo_model and hasattr(self.geo_model, 'cache') else 0
            },
            'dataset': {
                'loaded': self.dataset is not None,
                'records': len(self.dataset) if self.dataset is not None else 0,
                'states': self.dataset['STATE'].nunique() if self.dataset is not None else 0,
                'cities': self.dataset['CITY'].nunique() if self.dataset is not None else 0
            }
        }
        return info


def test_integrated_loader():
    """Test the integrated model loader"""
    print("="*70)
    print("INTEGRATED MODEL LOADER TEST")
    print("="*70)
    
    # Load models
    loader = IntegratedIPGeolocator()
    success = loader.load_models()
    
    if not success:
        print("\n\u26a0\ufe0f  Some models failed to load, but continuing with available models...")
    
    # Show model info
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    info = loader.get_model_info()
    
    print("\nAdvanced Model:")
    print(f"  Loaded: {info['advanced_model']['loaded']}")
    print(f"  Type: {info['advanced_model']['type']}")
    print(f"  Features: {info['advanced_model']['features']}")
    
    print("\nGeoModel:")
    print(f"  Loaded: {info['geo_model']['loaded']}")
    print(f"  Type: {info['geo_model']['type']}")
    
    # Test predictions
    print("\n" + "="*70)
    print("TEST PREDICTIONS")
    print("="*70)
    
    test_ip = "117.242.89.241"
    print(f"\nTesting IP: {test_ip}")
    
    # Test with network metrics (advanced regression model)
    network_metrics = {
        'ping_delay_time': 61.682,
        'last_router_delay': 2.451,
        'traceroute_steps': 7,
        'traceroute_time': 20.285,
        'total_delay': 30.26,
        'minimum_rtt': 25.56,
        'maximum_rtt': 40.766,
        'average_rtt': 34.229,
        'time_to_live': 61,
        'latitude': 9.9312,
        'longitude': 76.2673
    }
    
    result = loader.predict_hybrid(test_ip, network_metrics)
    
    print(f"\nPrimary Prediction:")
    if result['primary']:
        pred = result['primary']
        print(f"  Method: {pred['method']}")
        print(f"  State: {pred['state']}")
        print(f"  City: {pred['city']}")
        print(f"  Coordinates: ({pred['lat']:.4f}, {pred['lon']:.4f})")
        print(f"  Confidence: {pred['confidence']*100:.1f}%")
    
    if result['fallback']:
        print(f"\nFallback Prediction:")
        pred = result['fallback']
        print(f"  Method: {pred['method']}")
        print(f"  State: {pred['state']}")
        print(f"  City: {pred['city']}")
        print(f"  Coordinates: ({pred['lat']:.4f}, {pred['lon']:.4f})")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)

if __name__ == "__main__":
    test_integrated_loader()
