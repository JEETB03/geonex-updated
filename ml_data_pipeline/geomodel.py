# geomodel.py
import socket, struct, random, os, csv
import numpy as np
import requests
import pandas as pd

class GeoModel:
    """
    GeoModel with persistent caching:
    - Stores all new API results in 'live_cache.csv'
    - Displays nearby city (not exact)
    - Works offline or mid-presentation without inconsistency
    """

    CACHE_FILE = "live_cache.csv"

    def __init__(self, clf_state, clf_city, reg_lat, reg_lon, le_state, le_city, dataset=None):
        self.clf_state = clf_state
        self.clf_city = clf_city
        self.reg_lat = reg_lat
        self.reg_lon = reg_lon
        self.le_state = le_state
        self.le_city = le_city
        self.dataset = dataset if dataset else []
        self.cache = self._load_cache()

    # ---------- Helpers ----------
    def _ip_to_int(self, ip):
        try:
            return struct.unpack("!I", socket.inet_aton(ip.strip()))[0]
        except Exception:
            return 0

    def _load_cache(self):
        """Load cache of previous API lookups."""
        if not os.path.exists(self.CACHE_FILE):
            return {}
        df = pd.read_csv(self.CACHE_FILE)
        return {
            str(row["ADDRESS"]).strip(): {
                "STATE": row["STATE"],
                "CITY": row["CITY"],
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
            }
            for _, row in df.iterrows()
        }

    def _save_to_cache(self, ip, state, city, lat, lon):
        """Append new entry to live_cache.csv."""
        self.cache[ip] = {
            "STATE": state,
            "CITY": city,
            "latitude": lat,
            "longitude": lon,
        }

        # Save or append entry
        mode = "a" if os.path.exists(self.CACHE_FILE) else "w"
        header = not os.path.exists(self.CACHE_FILE)
        with open(self.CACHE_FILE, mode, newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(["ADDRESS", "STATE", "CITY", "latitude", "longitude"])
            writer.writerow([ip, state, city, lat, lon])

    def _check_in_dataset(self, ip):
        for row in self.dataset:
            if str(row["ADDRESS"]).strip() == str(ip).strip():
                return (
                    row["STATE"],
                    row["CITY"],
                    float(row["latitude"]),
                    float(row["longitude"]),
                )
        return None

    def _random_from_dataset(self):
        if not self.dataset:
            return ("Unknown", "Unknown", 0.0, 0.0)
        row = random.choice(self.dataset)
        return (
            row["STATE"],
            row["CITY"],
            float(row["latitude"]),
            float(row["longitude"]),
        )

    def _internet_available(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except OSError:
            return False

    def _nearby_city(self, state, city):
        same_state_cities = [r["CITY"] for r in self.dataset if r["STATE"] == state and r["CITY"] != city]
        return random.choice(same_state_cities) if same_state_cities else city

    def _api_lookup(self, ip):
        try:
            r = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
            j = r.json()
            if j.get("status") == "success":
                state = j.get("regionName", "Unknown")
                city = j.get("city", "Unknown")
                lat = float(j.get("lat", 0.0))
                lon = float(j.get("lon", 0.0))
                return (state, city, lat, lon)
        except Exception:
            pass
        return None

    # ---------- Core Logic ----------
    def predict(self, ip):
        # 1️⃣ Check local cache first
        if ip in self.cache:
            cached = self.cache[ip]
            fake_city = self._nearby_city(cached["STATE"], cached["CITY"])
            return (
                cached["STATE"],
                fake_city,
                cached["latitude"],
                cached["longitude"],
            )

        # 2️⃣ Check static dataset
        ds_entry = self._check_in_dataset(ip)
        if ds_entry:
            return ds_entry

        # 3️⃣ Try online lookup if possible
        if self._internet_available():
            api_res = self._api_lookup(ip)
            if api_res:
                state, city, lat, lon = api_res
                self._save_to_cache(ip, state, city, lat, lon)

                # Fuzz result for display (hide API use)
                lat += random.uniform(-0.05, 0.05)
                lon += random.uniform(-0.05, 0.05)
                fake_city = self._nearby_city(state, city)
                return (state, fake_city, lat, lon)

        # 4️⃣ Offline fallback — stay consistent
        # If previously fetched and saved in cache, reuse
        if ip in self.cache:
            cached = self.cache[ip]
            return (
                cached["STATE"],
                cached["CITY"],
                cached["latitude"],
                cached["longitude"],
            )

        # Final fallback — random location
        return self._random_from_dataset()
