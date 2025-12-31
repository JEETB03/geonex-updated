# 🌍 GeoNex: Advanced IP Intelligence & Geolocation

**GeoNex** is a state-of-the-art IP geolocation and network intelligence platform. It combines active network probing with advanced Machine Learning (XGBoost Regression) to predict IP locations with high precision, even for targets not in standard databases.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-green.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## ✨ Key Features

### 🖥️ Slick GUI v3.0
A modern, dark-themed interface built with `CustomTkinter` for a premium user experience.
- **Dashboard**: Interactive map view and real-time model status.
- **Deep Scan**: Active network probing (Ping, Traceroute, TTL) feeding into a hybrid ML model for pin-point accuracy.
- **Security Center**: comprehensive threat analysis including:
    - 🛡️ VPN / Proxy Detection (via VPNAPI.io)
    - 🧅 Tor Exit Node Identification
    - 🏢 Carrier-Grade NAT (CGNAT) Detection
- **Commercial Benchmark**: Real-time comparison against industry giants like **IP-API** and **IPInfo**.
- **Interactive API Console**: Built-in tool to test API endpoints directly from the settings page.

### 🧠 Advanced ML Core
- **Hybrid Engine**: Combines static database lookups with dynamic network metrics.
- **XGBoost Regression**: Predicts Latitude/Longitude based on millisecond-level latency patterns.
- **Auto-Learning**: Automatically loads the latest trained models (`advanced_loc_model.joblib`) on startup.

### 🚀 High-Performance API
- **FastAPI Backend**: Production-ready REST API.
- **Endpoints**:
    - `GET /api/geolocate?ip=8.8.8.8`: Get location and confidence.
    - `GET /api/cgnat?ip=8.8.8.8`: Detect ISP-level NAT.
    - `POST /api/analyze`: Full security and location report.
    - `POST /api/batch`: Process thousands of IPs in parallel.

---

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Linux / macOS / Windows

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/JEETB03/geonex-updated.git
   cd geonex-updated
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Models:**
   Ensure `advanced_loc_model.joblib` is present in the root directory.

---

## 🚀 Usage

### Running the GUI
Launch the modern interface with the provided script:
```bash
./run_slick_gui.sh
```

### Running the API Server
Start the high-performance backend:
```bash
python3 -m uvicorn api_server:app --port 8000 --reload
```
*Documentation available at: http://localhost:8000/docs*

---

## 📊 Benchmarking

GeoNex includes a unique **Benchmark Tab** that allows you to compare its accuracy against commercial services in real-time.

| Service | Method | Features |
|---------|--------|----------|
| **GeoNex (Local)** | AI + Active Probing | Free, Privacy-focused, No limits |
| **IP-API.com** | Database | Fast, Standard Accuracy |
| **IPInfo.io** | Database | Commercial Grade, Rate Limited |

---

## 📂 Project Structure

```
src/
├── slick_gui.py              # Main GUI Application
├── api_server.py             # FastAPI Backend
├── integrated_model_loader.py # Core ML Logic
├── advanced_loc_model.joblib # Trained Machine Learning Model
├── detect_cgnat.py           # CGNAT Detection Module
├── run_slick_gui.sh          # Launcher Script
└── requirements.txt          # Python Dependencies
```

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
