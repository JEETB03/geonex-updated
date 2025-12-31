#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoNex API Client
Client library for accessing GeoNex API from GUI or other applications

Usage:
    from api_client import GeoNexAPIClient
    
    client = GeoNexAPIClient("http://localhost:8000")
    result = client.geolocate("8.8.8.8")
"""

import requests
from typing import Dict, List, Optional, Any
import json


class GeoNexAPIClient:
    """
    Client for GeoNex API
    
    Provides easy access to all API endpoints with error handling
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 10):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API server (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 10)
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make HTTP request to API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout: API server not responding")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Connection error: Cannot reach API server at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            error_detail = "Unknown error"
            try:
                error_data = response.json()
                error_detail = error_data.get('detail', str(e))
            except:
                error_detail = str(e)
            raise Exception(f"API error: {error_detail}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health status
        
        Returns:
            Health status information
        """
        return self._make_request("GET", "/health")
    
    def geolocate(self, ip: str) -> Dict[str, Any]:
        """
        Geolocate an IP address
        
        Args:
            ip: IP address to geolocate
            
        Returns:
            Geolocation information including country, state, city, coordinates
        """
        return self._make_request("GET", "/api/geolocate", params={"ip": ip})
    
    def detect_cgnat(self, ip: str, skip_traceroute: bool = True) -> Dict[str, Any]:
        """
        Detect CGNAT for an IP address
        
        Args:
            ip: IP address to check
            skip_traceroute: Skip traceroute analysis (faster)
            
        Returns:
            CGNAT detection results
        """
        return self._make_request(
            "GET",
            "/api/cgnat",
            params={"ip": ip, "skip_traceroute": skip_traceroute}
        )
    
    def analyze(self, ip: str, include_cgnat: bool = True, include_security: bool = False) -> Dict[str, Any]:
        """
        Comprehensive IP analysis
        
        Args:
            ip: IP address to analyze
            include_cgnat: Include CGNAT detection
            include_security: Include security analysis
            
        Returns:
            Complete analysis results
        """
        data = {
            "ip": ip,
            "include_cgnat": include_cgnat,
            "include_security": include_security
        }
        return self._make_request("POST", "/api/analyze", json=data)
    
    def batch_analyze(self, ips: List[str], include_cgnat: bool = True) -> Dict[str, Any]:
        """
        Batch analyze multiple IP addresses
        
        Args:
            ips: List of IP addresses (max 50)
            include_cgnat: Include CGNAT detection
            
        Returns:
            Batch analysis results
        """
        if len(ips) > 50:
            raise ValueError("Maximum 50 IPs per batch request")
        
        data = {
            "ips": ips,
            "include_cgnat": include_cgnat
        }
        return self._make_request("POST", "/api/batch", json=data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models
        
        Returns:
            Model information
        """
        return self._make_request("GET", "/api/models/info")
    
    def test_connection(self) -> bool:
        """
        Test if API server is reachable
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            self.health_check()
            return True
        except:
            return False


# Example usage and testing
def test_api_client():
    """Test API client functionality"""
    print("="*70)
    print("Testing GeoNex API Client")
    print("="*70)
    
    # Initialize client
    client = GeoNexAPIClient("http://localhost:8000")
    
    # Test 1: Health check
    print("\n[TEST 1] Health Check...")
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"   Models Loaded: {health['models_loaded']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("\n⚠️  Make sure API server is running:")
        print("   python api_server.py")
        return False
    
    # Test 2: Geolocate IP
    print("\n[TEST 2] Geolocate IP...")
    try:
        result = client.geolocate("8.8.8.8")
        print(f"✅ IP: {result['ip']}")
        print(f"   Location: {result['city']}, {result['state']}, {result['country']}")
        print(f"   Coordinates: ({result['latitude']}, {result['longitude']})")
        print(f"   Confidence: {result['confidence']}")
    except Exception as e:
        print(f"❌ Geolocation failed: {e}")
    
    # Test 3: CGNAT Detection
    print("\n[TEST 3] CGNAT Detection...")
    try:
        result = client.detect_cgnat("100.64.0.1")
        print(f"✅ IP: {result['ip']}")
        print(f"   CGNAT Detected: {result['cgnat_detected']}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Score: {result['score']}/100")
    except Exception as e:
        print(f"❌ CGNAT detection failed: {e}")
    
    # Test 4: Comprehensive Analysis
    print("\n[TEST 4] Comprehensive Analysis...")
    try:
        result = client.analyze("117.242.89.241", include_cgnat=True)
        print(f"✅ IP: {result['ip']}")
        if 'geolocation' in result:
            geo = result['geolocation']
            print(f"   Location: {geo.get('city')}, {geo.get('state')}")
        if 'cgnat' in result:
            cgnat = result['cgnat']
            print(f"   CGNAT: {cgnat['detected']} (Confidence: {cgnat['confidence']})")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
    
    # Test 5: Batch Analysis
    print("\n[TEST 5] Batch Analysis...")
    try:
        ips = ["8.8.8.8", "1.1.1.1", "117.242.89.241"]
        result = client.batch_analyze(ips, include_cgnat=True)
        print(f"✅ Total: {result['total']}")
        print(f"   Successful: {result['successful']}")
        print(f"   Failed: {result['failed']}")
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
    
    print("\n" + "="*70)
    print("✅ API Client Tests Complete")
    print("="*70)
    
    return True


if __name__ == "__main__":
    test_api_client()
