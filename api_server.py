#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoNex API Server
REST API for IP Geolocation, CGNAT Detection, and Security Analysis

Features:
- IP Geolocation prediction
- CGNAT detection
- VPN/Proxy/Tor detection
- Multi-IP batch processing
- Model information
- Health checks

Usage:
    python api_server.py
    
    Or with custom host/port:
    python api_server.py --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import sys

# Import our modules
from integrated_model_loader import IntegratedIPGeolocator
from detect_cgnat import CGNATDetector

# Initialize FastAPI app
app = FastAPI(
    title="GeoNex API",
    description="Advanced IP Geolocation, CGNAT Detection, and Security Analysis API",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("🔄 Loading models...")
integrated_loader = IntegratedIPGeolocator()
cgnat_detector = CGNATDetector()

try:
    integrated_loader.load_models()
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Some models failed to load: {e}")

# Pydantic models for request/response
class IPRequest(BaseModel):
    ip: str = Field(..., description="IP address to analyze", example="8.8.8.8")
    include_cgnat: bool = Field(True, description="Include CGNAT detection")
    include_security: bool = Field(False, description="Include VPN/Security check (requires API key)")

class MultiIPRequest(BaseModel):
    ips: List[str] = Field(..., description="List of IP addresses", example=["8.8.8.8", "1.1.1.1"])
    include_cgnat: bool = Field(True, description="Include CGNAT detection")

class GeolocationResponse(BaseModel):
    ip: str
    country: str
    state: str
    city: str
    latitude: float
    longitude: float
    confidence: float
    method: str
    timestamp: str

class CGNATResponse(BaseModel):
    ip: str
    cgnat_detected: bool
    confidence: str
    score: int
    is_cgnat_range: bool
    is_private: bool
    is_loopback: bool
    reasons: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    version: str

# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "GeoNex API",
        "version": "2.0",
        "description": "Advanced IP Geolocation, CGNAT Detection, and Security Analysis",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "geolocate": "/api/geolocate",
            "cgnat": "/api/cgnat",
            "analyze": "/api/analyze",
            "batch": "/api/batch"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    model_info = integrated_loader.get_model_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "advanced_model": model_info['advanced_model']['loaded'],
            "geo_model": model_info['geo_model']['loaded'],
            "dataset": model_info['dataset']['loaded']
        },
        "version": "2.0"
    }

@app.get("/api/geolocate", tags=["Geolocation"])
async def geolocate_ip(
    ip: str = Query(..., description="IP address to geolocate", example="8.8.8.8")
):
    """
    Geolocate an IP address
    
    Returns location information including country, state, city, and coordinates.
    """
    try:
        # Use integrated loader for prediction
        result = integrated_loader.predict_hybrid(ip)
        
        if not result['primary']:
            raise HTTPException(status_code=404, detail="Could not geolocate IP address")
        
        primary = result['primary']
        
        return {
            "ip": ip,
            "country": primary.get('country', 'Unknown'),
            "state": primary.get('state', 'Unknown'),
            "city": primary.get('city', 'Unknown'),
            "latitude": primary.get('lat', 0.0),
            "longitude": primary.get('lon', 0.0),
            "confidence": primary.get('confidence', 0.0),
            "method": primary.get('method', 'Unknown'),
            "model_type": primary.get('model_type', 'Unknown'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geolocation error: {str(e)}")

@app.get("/api/cgnat", tags=["CGNAT Detection"])
async def detect_cgnat(
    ip: str = Query(..., description="IP address to check for CGNAT", example="100.64.0.1"),
    skip_traceroute: bool = Query(True, description="Skip traceroute (faster)")
):
    """
    Detect CGNAT (Carrier-Grade NAT) for an IP address
    
    Returns CGNAT detection results including confidence score and indicators.
    """
    try:
        analysis = cgnat_detector.analyze_cgnat(ip, skip_traceroute=skip_traceroute)
        
        if analysis.get('error'):
            raise HTTPException(status_code=400, detail=analysis['error'])
        
        return {
            "ip": analysis['ip'],
            "cgnat_detected": analysis['cgnat_detected'],
            "confidence": analysis['confidence'],
            "score": analysis.get('cgnat_score', 0),
            "is_cgnat_range": analysis['is_cgnat'],
            "is_private": analysis['is_private'],
            "is_loopback": analysis['is_loopback'],
            "local_ip": analysis['local_ip'],
            "reasons": analysis['reasons'],
            "port_results": analysis['port_results'],
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CGNAT detection error: {str(e)}")

@app.post("/api/analyze", tags=["Comprehensive Analysis"])
async def analyze_ip(request: IPRequest):
    """
    Comprehensive IP analysis
    
    Combines geolocation and CGNAT detection in a single request.
    """
    try:
        result = {
            "ip": request.ip,
            "timestamp": datetime.now().isoformat()
        }
        
        # Geolocation
        geo_result = integrated_loader.predict_hybrid(request.ip)
        if geo_result['primary']:
            primary = geo_result['primary']
            result['geolocation'] = {
                "country": primary.get('country', 'Unknown'),
                "state": primary.get('state', 'Unknown'),
                "city": primary.get('city', 'Unknown'),
                "latitude": primary.get('lat', 0.0),
                "longitude": primary.get('lon', 0.0),
                "confidence": primary.get('confidence', 0.0),
                "method": primary.get('method', 'Unknown')
            }
        else:
            result['geolocation'] = {"error": "Could not geolocate IP"}
        
        # CGNAT Detection
        if request.include_cgnat:
            cgnat_analysis = cgnat_detector.analyze_cgnat(request.ip, skip_traceroute=True)
            result['cgnat'] = {
                "detected": cgnat_analysis['cgnat_detected'],
                "confidence": cgnat_analysis['confidence'],
                "score": cgnat_analysis.get('cgnat_score', 0),
                "is_cgnat_range": cgnat_analysis['is_cgnat'],
                "is_private": cgnat_analysis['is_private'],
                "reasons": cgnat_analysis['reasons'][:3]  # Top 3 reasons
            }
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/api/batch", tags=["Batch Processing"])
async def batch_analyze(request: MultiIPRequest):
    """
    Batch IP analysis
    
    Analyze multiple IP addresses in a single request.
    Limited to 50 IPs per request.
    """
    if len(request.ips) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 IPs per batch request")
    
    try:
        results = []
        
        for ip in request.ips:
            ip_result = {
                "ip": ip,
                "status": "success"
            }
            
            try:
                # Geolocation
                geo_result = integrated_loader.predict_hybrid(ip)
                if geo_result['primary']:
                    primary = geo_result['primary']
                    ip_result['geolocation'] = {
                        "state": primary.get('state', 'Unknown'),
                        "city": primary.get('city', 'Unknown'),
                        "latitude": primary.get('lat', 0.0),
                        "longitude": primary.get('lon', 0.0),
                        "confidence": primary.get('confidence', 0.0)
                    }
                
                # CGNAT Detection
                if request.include_cgnat:
                    cgnat_analysis = cgnat_detector.analyze_cgnat(ip, skip_traceroute=True)
                    ip_result['cgnat'] = {
                        "detected": cgnat_analysis['cgnat_detected'],
                        "confidence": cgnat_analysis['confidence'],
                        "score": cgnat_analysis.get('cgnat_score', 0)
                    }
            
            except Exception as e:
                ip_result['status'] = "error"
                ip_result['error'] = str(e)
            
            results.append(ip_result)
        
        return {
            "total": len(request.ips),
            "successful": len([r for r in results if r['status'] == 'success']),
            "failed": len([r for r in results if r['status'] == 'error']),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

@app.get("/api/models/info", tags=["Models"])
async def get_model_info():
    """Get information about loaded models"""
    try:
        info = integrated_loader.get_model_info()
        return {
            "advanced_model": info['advanced_model'],
            "geo_model": info['geo_model'],
            "dataset": info['dataset'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "detail": "The requested endpoint does not exist",
        "available_endpoints": ["/docs", "/health", "/api/geolocate", "/api/cgnat", "/api/analyze"]
    }

# Main entry point
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GeoNex API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1, use 0.0.0.0 for all interfaces)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("="*70)
    print("🌍 GeoNex API Server v2.0")
    print("="*70)
    print(f"\n📡 Starting server on http://{args.host}:{args.port}")
    print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc: http://{args.host}:{args.port}/redoc")
    print(f"🏥 Health Check: http://{args.host}:{args.port}/health")
    print("\n" + "="*70)
    print("Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
