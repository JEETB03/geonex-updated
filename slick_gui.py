#!/usr/bin/env python3
"""
GeoNex Slick GUI v3.0
Powered by CustomTkinter for a modern, high-performance interface.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import os
import sys
import glob
import pickle
import joblib
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Local imports (reuse existing logic)
from integrated_model_loader import IntegratedIPGeolocator
from detect_cgnat import CGNATDetector
from api_client import GeoNexAPIClient

try:
    import tkintermapview
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False
    print("⚠️  tkintermapview not available. Map features will be disabled.")

# Helper for collecting network metrics (copied from enhanced_gui_v2.py)
def gather_network_metrics(ip_address):
    """Gather actual network metrics for an IP address"""
    try:
        import subprocess
        import platform
        import re
        
        metrics = {}
        
        # Ping the IP to get RTT metrics
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, '4', ip_address]
        
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        output = result.stdout
        
        # Parse ping results
        times = re.findall(r'time[=<](\d+\.?\d*)', output)
        if times:
            times = [float(t) for t in times]
            metrics['minimum_rtt'] = min(times)
            metrics['maximum_rtt'] = max(times)
            metrics['average_rtt'] = sum(times) / len(times)
            metrics['ping_delay_time'] = metrics['average_rtt']
        else:
            # Use defaults if parsing fails
            metrics['minimum_rtt'] = 30.0
            metrics['maximum_rtt'] = 150.0
            metrics['average_rtt'] = 80.0
            metrics['ping_delay_time'] = 80.0
        
        # Extract TTL
        ttl_match = re.search(r'(?:TTL|ttl)=(\d+)', output)
        metrics['time_to_live'] = int(ttl_match.group(1)) if ttl_match else 64
        
        # Estimate other metrics based on RTT
        metrics['last_router_delay'] = metrics['average_rtt'] * 0.3
        metrics['traceroute_steps'] = 10  # Estimated
        metrics['traceroute_time'] = metrics['average_rtt'] * 0.5
        metrics['total_delay'] = metrics['average_rtt'] * 1.2
        
        return metrics
        
    except Exception as e:
        print(f"Error gathering network metrics: {e}")
        return None

class SlickGeoNexApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # System Settings
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        # Window Config
        self.title("GeoNex Slick v3.0 - Advanced IP Intelligence")
        self.geometry("1400x900")
        
        # Try to maximize
        try:
            self.state('zoomed')
        except:
            self.attributes('-zoomed', True)

        # Data State
        self.integrated_loader = IntegratedIPGeolocator()
        self.cgnat_detector = CGNATDetector()
        self.api_client = None
        self.api_mode = False
        self.api_url = "http://localhost:8000"
        self.tested_ips = []
        
        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.create_sidebar()

        # 2. Main Area (Pages)
        self.create_pages()
        
        # 3. Initialize First Page
        self.show_page("dashboard")

        # 4. Auto-load Model
        self.after(100, self.auto_load_model)

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(6, weight=1)

        # Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar, 
            text="🌍 GeoNex\nSlick v3.0", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Nav Buttons
        self.btn_dashboard = self.create_nav_btn("📊 Dashboard", "dashboard", 1)
        self.btn_deepscan = self.create_nav_btn("🔍 Deep Scan", "deep_scan", 2)
        self.btn_security = self.create_nav_btn("🛡️ Security", "security", 3)
        self.btn_compare = self.create_nav_btn("⚖️ Benchmark", "benchmark", 4)
        self.btn_settings = self.create_nav_btn("⚙️ Settings", "settings", 5)

        # Status at bottom
        self.status_label = ctk.CTkLabel(
            self.sidebar, 
            text="Initializing...", 
            text_color="gray",
            wraplength=180
        )
        self.status_label.grid(row=7, column=0, padx=20, pady=20)

    def create_nav_btn(self, text, page_name, row):
        btn = ctk.CTkButton(
            self.sidebar, 
            text=text, 
            command=lambda: self.show_page(page_name),
            fg_color="transparent", 
            text_color=("gray10", "gray90"), 
            hover_color=("gray70", "gray30"),
            anchor="w",
            height=40
        )
        btn.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        return btn

    def create_pages(self):
        # Container for pages
        self.pages_container = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.pages_container.grid(row=0, column=1, sticky="nsew")
        self.pages_container.grid_rowconfigure(0, weight=1)
        self.pages_container.grid_columnconfigure(0, weight=1)

        self.pages = {}
        
        # --- Dashboard Page ---
        self.pages["dashboard"] = self.create_dashboard_page(self.pages_container)
        
        # --- Deep Scan Page ---
        self.pages["deep_scan"] = self.create_deep_scan_page(self.pages_container)
        
        # --- Security Page ---
        self.pages["security"] = self.create_security_page(self.pages_container)

        # --- Benchmark Page ---
        self.pages["benchmark"] = self.create_benchmark_page(self.pages_container)

        # --- Settings Page ---
        self.pages["settings"] = self.create_settings_page(self.pages_container)

    def create_dashboard_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(frame, height=60, corner_radius=10)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(header, text="Operations Dashboard", font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=20, pady=10)
        
        self.model_status_lbl = ctk.CTkLabel(header, text="● Model: Loading...", text_color="orange")
        self.model_status_lbl.pack(side="right", padx=20)

        # Map Area
        map_frame = ctk.CTkFrame(frame, corner_radius=10)
        map_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        if MAP_AVAILABLE:
            self.map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=10)
            self.map_widget.pack(fill="both", expand=True)
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}", max_zoom=22)
            self.map_widget.set_position(20.5937, 78.9629) # India
            self.map_widget.set_zoom(5)
        else:
            ctk.CTkLabel(map_frame, text="Map View Unavailable").pack(expand=True)

        return frame

    def create_deep_scan_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=2)
        frame.grid_rowconfigure(0, weight=1)

        # Left Panel (Input)
        left_panel = ctk.CTkFrame(frame, corner_radius=10)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        ctk.CTkLabel(left_panel, text="Deep Scan Target", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        self.scan_ip_entry = ctk.CTkEntry(left_panel, placeholder_text="Enter IP Address (e.g. 8.8.8.8)")
        self.scan_ip_entry.pack(fill="x", padx=20, pady=10)
        
        self.btn_scan = ctk.CTkButton(left_panel, text="Initiate Deep Scan", command=self.run_deep_scan)
        self.btn_scan.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(left_panel, text="Scan Options", font=ctk.CTkFont(weight="bold")).pack(pady=(20,10), anchor="w", padx=20)
        self.chk_metrics = ctk.CTkCheckBox(left_panel, text="Active Network Probing")
        self.chk_metrics.select()
        self.chk_metrics.pack(anchor="w", padx=20, pady=5)

        # Right Panel (Results)
        right_panel = ctk.CTkFrame(frame, corner_radius=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        
        ctk.CTkLabel(right_panel, text="Analysis Results", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=20)
        
        self.scan_results_box = ctk.CTkTextbox(right_panel, font=ctk.CTkFont(family="Consolas", size=14))
        self.scan_results_box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        self.scan_results_box.insert("0.0", "Ready to scan. Enter an IP address to begin.")
        self.scan_results_box.configure(state="disabled")

        return frame

    def create_security_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        # Top Bar
        top_frame = ctk.CTkFrame(frame)
        top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        
        ctk.CTkLabel(top_frame, text="Security Intelligence Center", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=20, pady=15)
        
        self.sec_ip_entry = ctk.CTkEntry(top_frame, placeholder_text="Enter IP", width=200)
        self.sec_ip_entry.pack(side="left", padx=10)
        
        self.btn_sec_check = ctk.CTkButton(top_frame, text="Check Threat Level", command=self.run_security_check, fg_color="#ef4444", hover_color="#b91c1c")
        self.btn_sec_check.pack(side="left", padx=10)

        # Main Info Area
        self.sec_results_box = ctk.CTkTextbox(frame, font=ctk.CTkFont(family="Consolas", size=14))
        self.sec_results_box.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        welcome_msg = """
🔒 SECURITY INTELLIGENCE
------------------------
• VPN / Proxy Detection
• Tor Exit Node Identification
• Carrier-Grade NAT (CGNAT) Analysis
• IP Reputation & Risk Assessment

Enter an IP address to begin threat analysis.
"""
        self.sec_results_box.insert("0.0", welcome_msg)
        self.sec_results_box.configure(state="disabled")

        return frame

    def create_benchmark_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        
        # Top Bar
        top_frame = ctk.CTkFrame(frame)
        top_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        
        ctk.CTkLabel(top_frame, text="Commercial Benchmarking", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=20, pady=15)
        
        self.bench_ip_entry = ctk.CTkEntry(top_frame, placeholder_text="Enter IP for Comparison", width=250)
        self.bench_ip_entry.pack(side="left", padx=10)
        
        self.btn_benchmark = ctk.CTkButton(top_frame, text="Run Comparison", command=self.run_benchmark)
        self.btn_benchmark.pack(side="left", padx=10)
        
        # Results Area (Scrollable)
        self.bench_scroll = ctk.CTkScrollableFrame(frame, label_text="Service Comparison Results")
        self.bench_scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        
        return frame

    def run_benchmark(self):
        ip = self.bench_ip_entry.get().strip()
        if not ip: return
        
        # Clear previous
        for widget in self.bench_scroll.winfo_children():
            widget.destroy()
            
        status_lbl = ctk.CTkLabel(self.bench_scroll, text="🔄 Fetching data from commercial providers...", font=ctk.CTkFont(size=16))
        status_lbl.pack(pady=50)
        
        self.btn_benchmark.configure(state="disabled")
        
        def _bench_thread():
            results = {}
            try:
                # 1. GeoNex (Our Model)
                t_start = time.time()
                metrics = gather_network_metrics(ip) # Use active probing for fair comparison
                pred = self.integrated_loader.predict_hybrid(ip, metrics)
                t_end = (time.time() - t_start) * 1000
                
                if pred and pred['primary']:
                    p = pred['primary']
                    results['GeoNex (Local AI)'] = {
                        'status': 'success',
                        'country': p.get('country'),
                        'state': p.get('state'),
                        'city': p.get('city'),
                        'lat': p.get('lat'),
                        'lon': p.get('lon'),
                        'confidence': f"{p.get('confidence', 0)*100:.1f}%",
                        'latency': f"{t_end:.0f}ms",
                        'color': '#4f46e5'
                    }
                else:
                    results['GeoNex (Local AI)'] = {'status': 'error', 'message': 'Prediction failed', 'color': '#4f46e5'}

                # 2. IP-API.com
                try:
                    t_start = time.time()
                    resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
                    t_end = (time.time() - t_start) * 1000
                    if resp.get('status') == 'success':
                        results['IP-API.com'] = {
                            'status': 'success',
                            'country': resp.get('country'),
                            'state': resp.get('regionName'),
                            'city': resp.get('city'),
                            'lat': resp.get('lat'),
                            'lon': resp.get('lon'),
                            'isp': resp.get('isp'),
                            'latency': f"{t_end:.0f}ms",
                            'color': '#059669'
                        }
                    else:
                         results['IP-API.com'] = {'status': 'error', 'message': resp.get('message'), 'color': '#059669'}
                except Exception as e:
                    results['IP-API.com'] = {'status': 'error', 'message': str(e), 'color': '#059669'}

                # 3. IPInfo.io
                try:
                    t_start = time.time()
                    resp = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5).json()
                    t_end = (time.time() - t_start) * 1000
                    loc = resp.get('loc', '0,0').split(',')
                    results['IPInfo.io'] = {
                        'status': 'success',
                        'country': resp.get('country'),
                        'state': resp.get('region'),
                        'city': resp.get('city'),
                        'lat': float(loc[0]) if len(loc) > 0 else 0,
                        'lon': float(loc[1]) if len(loc) > 1 else 0,
                        'org': resp.get('org'),
                        'latency': f"{t_end:.0f}ms",
                        'color': '#d946ef'
                    }
                except Exception as e:
                    results['IPInfo.io'] = {'status': 'error', 'message': str(e), 'color': '#d946ef'}

                # Display Results
                status_lbl.destroy()
                self.display_benchmark_results(results, ip)
                
            except Exception as e:
                print(f"Benchmark Error: {e}")
            finally:
                self.btn_benchmark.configure(state="normal")
        
        threading.Thread(target=_bench_thread, daemon=True).start()

    def display_benchmark_results(self, results, ip):
        # Summary Header
        ctk.CTkLabel(self.bench_scroll, text=f"Benchmark Results for {ip}", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0,20))
        
        for name, data in results.items():
            card = ctk.CTkFrame(self.bench_scroll, fg_color=("gray85", "gray17"))
            card.pack(fill="x", padx=10, pady=10)
            
            # Header
            header = ctk.CTkFrame(card, height=40, fg_color=data.get('color', 'gray'))
            header.pack(fill="x")
            ctk.CTkLabel(header, text=name, text_color="white", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=15, pady=5)
            
            if data['status'] == 'success':
                ctk.CTkLabel(header, text=f"⏱ {data['latency']}", text_color="white").pack(side="right", padx=15)
                
                # Content Grid
                content = ctk.CTkFrame(card, fg_color="transparent")
                content.pack(fill="x", padx=15, pady=10)
                
                # Row 1
                ctk.CTkLabel(content, text="Location:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=5)
                ctk.CTkLabel(content, text=f"{data.get('city')}, {data.get('state')}, {data.get('country')}").grid(row=0, column=1, sticky="w", padx=5)
                
                # Row 2
                ctk.CTkLabel(content, text="Coordinates:", font=ctk.CTkFont(weight="bold")).grid(row=1, column=0, sticky="w", padx=5)
                ctk.CTkLabel(content, text=f"{data.get('lat')}, {data.get('lon')}").grid(row=1, column=1, sticky="w", padx=5)
                
                # Row 3 (Extras)
                extras = []
                if data.get('confidence'): extras.append(f"Confidence: {data['confidence']}")
                if data.get('isp'): extras.append(f"ISP: {data['isp']}")
                if data.get('org'): extras.append(f"Org: {data['org']}")
                
                if extras:
                    ctk.CTkLabel(content, text="Details:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", padx=5)
                    ctk.CTkLabel(content, text=" | ".join(extras)).grid(row=2, column=1, sticky="w", padx=5)
            else:
                ctk.CTkLabel(card, text=f"❌ Failed: {data.get('message')}", text_color="#ef4444").pack(pady=20)

    def create_settings_page(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        
        # --- Left Column: Config & Docs ---
        left_col = ctk.CTkFrame(frame, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # 1. System Config
        config_frame = ctk.CTkFrame(left_col)
        config_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(config_frame, text="System Configuration", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10, padx=15, anchor="w")
        
        self.chk_api_mode = ctk.CTkCheckBox(config_frame, text="Enable API Mode (Remote Server)", command=self.toggle_api_mode)
        self.chk_api_mode.pack(pady=5, padx=15, anchor="w")
        
        ctk.CTkLabel(config_frame, text="Server URL:").pack(pady=(5,0), padx=15, anchor="w")
        self.api_url_entry = ctk.CTkEntry(config_frame)
        self.api_url_entry.insert(0, self.api_url)
        self.api_url_entry.pack(pady=5, padx=15, fill="x")
        
        # 2. API Documentation
        docs_frame = ctk.CTkScrollableFrame(left_col, label_text="API Endpoints Reference")
        docs_frame.pack(fill="both", expand=True)
        
        endpoints = [
            ("GET /health", "Check server status & loaded models"),
            ("GET /api/geolocate?ip={ip}", "Get location for a single IP"),
            ("GET /api/cgnat?ip={ip}", "Detect CGNAT status"),
            ("POST /api/analyze", "Full analysis (Geo + Security)"),
            ("POST /api/batch", "Batch process multiple IPs")
        ]
        
        for ep, desc in endpoints:
            ef = ctk.CTkFrame(docs_frame, fg_color=("gray85", "gray20"))
            ef.pack(fill="x", pady=5)
            ctk.CTkLabel(ef, text=ep, font=ctk.CTkFont(family="Consolas", weight="bold")).pack(anchor="w", padx=10, pady=(5,0))
            ctk.CTkLabel(ef, text=desc, text_color="gray").pack(anchor="w", padx=10, pady=(0,5))

        # --- Right Column: Test Console ---
        right_col = ctk.CTkFrame(frame)
        right_col.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        ctk.CTkLabel(right_col, text="Interactive API Console", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
        
        # Endpoint Selector
        self.console_ep_var = ctk.StringVar(value="/api/geolocate")
        ctk.CTkOptionMenu(right_col, variable=self.console_ep_var, values=[
            "/health", "/api/geolocate", "/api/cgnat", "/api/analyze"
        ]).pack(pady=10)
        
        # Params Input
        ctk.CTkLabel(right_col, text="Parameters (JSON/Query):").pack(anchor="w", padx=20)
        self.console_input = ctk.CTkTextbox(right_col, height=100, font=ctk.CTkFont(family="Consolas"))
        self.console_input.insert("0.0", '{"ip": "8.8.8.8"}')
        self.console_input.pack(fill="x", padx=20, pady=5)
        
        # Run Button
        ctk.CTkButton(right_col, text="▶ Make Request", command=self.run_console_test).pack(pady=10)
        
        # Output
        ctk.CTkLabel(right_col, text="Response:").pack(anchor="w", padx=20)
        self.console_output = ctk.CTkTextbox(right_col, font=ctk.CTkFont(family="Consolas"), text_color="#10b981")
        self.console_output.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        return frame

    def run_console_test(self):
        endpoint = self.console_ep_var.get()
        base_url = self.api_url_entry.get().rstrip('/')
        url = f"{base_url}{endpoint}"
        
        # Parse params
        try:
            params = json.loads(self.console_input.get("0.0", "end"))
        except:
            params = {}
            
        self.console_output.delete("0.0", "end")
        self.console_output.insert("0.0", f"Requesting {url}...\n")
        
        def _req():
            try:
                if endpoint in ["/api/analyze", "/api/batch"]:
                    resp = requests.post(url, json=params, timeout=5)
                else:
                    resp = requests.get(url, params=params, timeout=5)
                
                out = json.dumps(resp.json(), indent=2)
                self.console_output.delete("0.0", "end")
                self.console_output.insert("0.0", out)
            except Exception as e:
                self.console_output.insert("end", f"\nError: {e}")
        
        threading.Thread(target=_req, daemon=True).start()
    
    def run_security_check(self):
        ip = self.sec_ip_entry.get().strip()
        if not ip: return
        
        self.update_sec_results(f"Analyzing threats for {ip}...")
        
        def _sec_thread():
            try:
                # 1. CGNAT Check
                cgnat = self.cgnat_detector.analyze_cgnat(ip, skip_traceroute=True)
                
                # 2. VPNAPI.io Check
                api_key = "b505e605ec2d4b7cb00b05c84963b609"
                url = f"https://vpnapi.io/api/{ip}?key={api_key}"
                vpn_data = {}
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        vpn_data = resp.json()
                except:
                    pass
                
                # 3. Format Report
                cgnat_status = "DETECTED" if cgnat['cgnat_detected'] else "Not Detected"
                security = vpn_data.get('security', {})
                
                report = f"""
🛡️ THREAT ANALYSIS REPORT
═══════════════════════════════════════════
TARGET: {ip}

🚨 THREAT INDICATORS:
   VPN:   {"YES 🔴" if security.get('vpn') else "NO 🟢"}
   Proxy: {"YES 🔴" if security.get('proxy') else "NO 🟢"}
   Tor:   {"YES 🔴" if security.get('tor') else "NO 🟢"}
   Relay: {"YES 🔴" if security.get('relay') else "NO 🟢"}

🌐 NETWORK INFRASTRUCTURE:
   Carrier-Grade NAT: {cgnat_status}
   Confidence:        {cgnat.get('confidence', 'N/A')}
   Private IP:        {"YES" if cgnat.get('is_private') else "NO"}
   
📍 LOCATION INTELLIGENCE (External):
   City:    {vpn_data.get('location', {}).get('city', 'N/A')}
   Country: {vpn_data.get('location', {}).get('country', 'N/A')}
   Network: {vpn_data.get('network', {}).get('autonomous_system_organization', 'N/A')}

"""             
                self.update_sec_results(report, append=False)
                
            except Exception as e:
                self.update_sec_results(f"Error: {e}")
        
        threading.Thread(target=_sec_thread, daemon=True).start()

    def update_sec_results(self, text, append=True):
        self.sec_results_box.configure(state="normal")
        if not append:
            self.sec_results_box.delete("0.0", "end")
        self.sec_results_box.insert("end", text)
        self.sec_results_box.see("end")
        self.sec_results_box.configure(state="disabled")

    def toggle_api_mode(self):
        self.api_mode = self.chk_api_mode.get()
        state = "Enabled" if self.api_mode else "Disabled"
        self.update_status(f"API Mode {state}")

    def test_api_connection(self):
        url = self.api_url_entry.get()
        self.api_status_label.configure(text="Testing...", text_color="orange")
        
        def _test():
            try:
                client = GeoNexAPIClient(url)
                health = client.health_check()
                self.api_status_label.configure(text="● Connected", text_color="#10b981")
                messagebox.showinfo("Success", f"Connected to API v{health.get('version')}")
            except Exception as e:
                self.api_status_label.configure(text="● Failed", text_color="#ef4444")
                messagebox.showerror("Connection Failed", str(e))
        
        threading.Thread(target=_test, daemon=True).start()

    def show_page(self, page_name):
        # Hide all
        for page in self.pages.values():
            page.grid_forget()
        
        # Show selected
        self.pages[page_name].grid(row=0, column=0, sticky="nsew")
        
        # Update buttons
        buttons = {
            "dashboard": self.btn_dashboard,
            "deep_scan": self.btn_deepscan,
            "security": self.btn_security,
            "benchmark": self.btn_compare,
            "settings": self.btn_settings
        }
        
        for name, btn in buttons.items():
            if name == page_name:
                btn.configure(fg_color=("gray75", "gray25"))
            else:
                btn.configure(fg_color="transparent")

    def auto_load_model(self):
        """Load ML models in background"""
        def _load():
            try:
                success = self.integrated_loader.load_models()
                if success:
                    self.update_status("Models Loaded Successfully")
                    self.model_status_lbl.configure(text="● Model: Active", text_color="#10b981") # Green
                else:
                    self.update_status("Model Load Failed (Partial)")
                    self.model_status_lbl.configure(text="● Model: Partial", text_color="orange")
            except Exception as e:
                print(f"Load Error: {e}")
                self.update_status(f"Error: {e}")
        
        threading.Thread(target=_load, daemon=True).start()

    def update_status(self, text):
        self.status_label.configure(text=text)

    def run_deep_scan(self):
        ip = self.scan_ip_entry.get().strip()
        if not ip:
            messagebox.showwarning("Input Error", "Please enter a valid IP address")
            return

        self.update_results("Initializing scan for " + ip + "...")
        self.btn_scan.configure(state="disabled")
        
        def _scan_thread():
            try:
                # 1. Gather Metrics
                if self.chk_metrics.get():
                    self.update_results(f"Probing network metrics for {ip}...\n(Ping, Traceroute, TTL Analysis)\n")
                    metrics = gather_network_metrics(ip)
                else:
                    metrics = None

                # 2. Predict
                self.update_results("\nRunning integrated ML models...\n")
                result = self.integrated_loader.predict_hybrid(ip, metrics)
                
                # 3. Format Output
                output = f"""
╔══════════════════════════════════════════════════╗
║              DEEP SCAN REPORT                    ║
╚══════════════════════════════════════════════════╝

🎯 TARGET: {ip}

"""
                if result['primary']:
                    p = result['primary']
                    output += f"""📍 PREDICTED LOCATION:
   Country:    {p.get('country', 'Unknown')}
   State:      {p.get('state', 'Unknown')}
   City:       {p.get('city', 'Unknown')}
   Confidence: {p.get('confidence', 0)*100:.1f}%

🛠️  METHOD: {p.get('method', 'N/A')}
"""
                
                if metrics:
                     output += f"""
📊 NETWORK TELEMETRY:
   Latency (Avg): {metrics.get('average_rtt', 0):.1f} ms
   Jitter:        {metrics.get('maximum_rtt', 0) - metrics.get('minimum_rtt', 0):.1f} ms
   Hops (Est):    {metrics.get('traceroute_steps', 0)}
   TTL:           {metrics.get('time_to_live', 0)}
"""

                self.update_results(output, append=False)
                
                # Update Map
                if MAP_AVAILABLE and result['primary']:
                    lat = result['primary'].get('lat', 0)
                    lon = result['primary'].get('lon', 0)
                    if lat != 0 and lon != 0:
                        self.map_widget.set_position(lat, lon)
                        self.map_widget.set_zoom(10)
                        self.map_widget.delete_all_marker()
                        self.map_widget.set_marker(lat, lon, text=f"{ip}\n{p.get('city')}")

            except Exception as e:
                self.update_results(f"\n❌ SCAN FAILED: {str(e)}")
            finally:
                self.btn_scan.configure(state="normal")

        threading.Thread(target=_scan_thread, daemon=True).start()

    def update_results(self, text, append=True):
        self.scan_results_box.configure(state="normal")
        if not append:
            self.scan_results_box.delete("0.0", "end")
        
        self.scan_results_box.insert("end", text)
        self.scan_results_box.see("end")
        self.scan_results_box.configure(state="disabled")

if __name__ == "__main__":
    app = SlickGeoNexApp()
    app.mainloop()
