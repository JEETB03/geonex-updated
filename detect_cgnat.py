#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGNAT Detection Module
Active CGNAT detection tool for IP geolocation system

Features:
- Detects Carrier-Grade NAT (CGNAT) / Shared Address Space
- Checks for private and shared IP ranges (100.64.0.0/10)
- Performs UDP traceroute to detect NAT in path
- Tests TCP port connectivity
- Provides heuristic analysis

Note: Traceroute requires elevated privileges (root/admin) on most systems
"""

import socket
import ipaddress
import struct
import time
from typing import Optional, List, Tuple, Dict


class CGNATDetector:
    """
    CGNAT Detection System
    Detects Carrier-Grade NAT and network address translation
    """
    
    # Shared address space for CGNAT (RFC 6598)
    CGNAT_NETWORK = ipaddress.ip_network("100.64.0.0/10")
    
    # Common ports to test
    DEFAULT_TEST_PORTS = [22, 80, 443, 8080, 3389]
    
    def __init__(self):
        self.has_raw_socket_permission = self._check_raw_socket_permission()
    
    def _check_raw_socket_permission(self) -> bool:
        """Check if we have permission to create raw sockets"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            s.close()
            return True
        except (PermissionError, OSError):
            return False
    
    def resolve_host(self, host: str) -> Optional[str]:
        """Resolve hostname to IP address"""
        try:
            return socket.gethostbyname(host)
        except Exception as e:
            return None
    
    def is_private_or_shared(self, ip_str: str) -> Tuple[bool, bool, bool]:
        """
        Check if IP is private, shared (CGNAT), or loopback
        
        Returns:
            (is_private, is_cgnat, is_loopback)
        """
        try:
            ip = ipaddress.ip_address(ip_str)
            is_private = ip.is_private
            is_cgnat = ip in self.CGNAT_NETWORK
            is_loopback = ip.is_loopback
            return (is_private, is_cgnat, is_loopback)
        except Exception:
            return (False, False, False)
    
    def get_local_ip_for_dest(self, dest_ip: str) -> str:
        """Get local IP address used to reach destination"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1.0)
            s.connect((dest_ip, 33434))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "0.0.0.0"
    
    def check_tcp_ports(self, dest_ip: str, ports: List[int] = None, timeout: float = 1.0) -> List[Tuple[int, bool]]:
        """
        Check if TCP ports are open/reachable
        
        Returns:
            List of (port, is_open) tuples
        """
        if ports is None:
            ports = self.DEFAULT_TEST_PORTS
        
        results = []
        for port in ports:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect((dest_ip, port))
                s.close()
                results.append((port, True))
            except Exception:
                results.append((port, False))
        
        return results
    
    def simple_traceroute(self, dest_ip: str, max_hops: int = 15) -> List[Tuple[int, Optional[str]]]:
        """
        Simplified traceroute using UDP with increasing TTL
        
        Note: Requires raw socket permissions for ICMP
        Returns: List of (ttl, hop_ip) tuples
        """
        if not self.has_raw_socket_permission:
            return []
        
        results = []
        dest_addr = dest_ip
        
        try:
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            recv_sock.settimeout(2.0)
        except (PermissionError, OSError):
            return []
        
        for ttl in range(1, max_hops + 1):
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            send_sock.setsockopt(socket.SOL_IP, socket.IP_TTL, ttl)
            
            port = 33434 + ttl
            hop_ip = None
            
            try:
                send_sock.sendto(b'', (dest_addr, port))
                data, addr = recv_sock.recvfrom(4096)
                hop_ip = addr[0]
            except socket.timeout:
                hop_ip = None
            except Exception:
                hop_ip = None
            finally:
                send_sock.close()
            
            results.append((ttl, hop_ip))
            
            if hop_ip == dest_addr:
                break
        
        recv_sock.close()
        return results
    
    def analyze_cgnat(self, target: str, skip_traceroute: bool = False) -> Dict:
        """
        Comprehensive CGNAT analysis
        
        Returns:
            Dictionary with analysis results
        """
        result = {
            'target': target,
            'ip': None,
            'resolved': False,
            'is_private': False,
            'is_cgnat': False,
            'is_loopback': False,
            'local_ip': '0.0.0.0',
            'traceroute': [],
            'port_results': [],
            'cgnat_detected': False,
            'confidence': 'Unknown',
            'reasons': [],
            'error': None
        }
        
        # Resolve hostname
        ip = self.resolve_host(target)
        if not ip:
            result['error'] = f"Could not resolve hostname: {target}"
            return result
        
        result['ip'] = ip
        result['resolved'] = True
        
        # Check IP type
        is_private, is_cgnat, is_loopback = self.is_private_or_shared(ip)
        result['is_private'] = is_private
        result['is_cgnat'] = is_cgnat
        result['is_loopback'] = is_loopback
        
        # Get local IP
        result['local_ip'] = self.get_local_ip_for_dest(ip)
        
        # Port check
        result['port_results'] = self.check_tcp_ports(ip)
        
        # Traceroute (if permitted and not skipped)
        if not skip_traceroute and self.has_raw_socket_permission:
            result['traceroute'] = self.simple_traceroute(ip, max_hops=15)
        
        # Analyze results
        reasons = []
        cgnat_score = 0
        
        # Check 1: CGNAT address space
        if is_cgnat:
            reasons.append("✓ IP is in 100.64.0.0/10 (RFC 6598 Shared Address Space)")
            cgnat_score += 50
            result['cgnat_detected'] = True
        
        # Check 2: Private IP
        if is_private and not is_loopback:
            reasons.append("✓ IP is in private address space (RFC 1918)")
            cgnat_score += 20
        
        # Check 3: Loopback
        if is_loopback:
            reasons.append("• IP is loopback address (localhost)")
            cgnat_score = 0
        
        # Check 4: Traceroute analysis
        if result['traceroute']:
            private_hops = 0
            cgnat_hops = 0
            
            for ttl, hop in result['traceroute']:
                if hop:
                    try:
                        hop_ip = ipaddress.ip_address(hop)
                        if hop_ip.is_private:
                            private_hops += 1
                        if hop_ip in self.CGNAT_NETWORK:
                            cgnat_hops += 1
                    except Exception:
                        pass
            
            if cgnat_hops > 0:
                reasons.append(f"✓ Traceroute shows {cgnat_hops} CGNAT hop(s) in path")
                cgnat_score += 30
            
            if private_hops > 0:
                reasons.append(f"✓ Traceroute shows {private_hops} private hop(s) in path")
                cgnat_score += 10
        
        # Check 5: Port accessibility
        closed_ports = sum(1 for _, is_open in result['port_results'] if not is_open)
        total_ports = len(result['port_results'])
        
        if total_ports > 0 and closed_ports == total_ports:
            reasons.append("✓ All tested ports are closed/filtered (NAT blocking inbound)")
            cgnat_score += 15
        elif total_ports > 0 and closed_ports > total_ports * 0.7:
            reasons.append(f"• Most ports ({closed_ports}/{total_ports}) are closed/filtered")
            cgnat_score += 5
        
        # Determine confidence
        if cgnat_score >= 50:
            result['confidence'] = 'High'
            result['cgnat_detected'] = True
        elif cgnat_score >= 30:
            result['confidence'] = 'Medium'
            result['cgnat_detected'] = True
        elif cgnat_score >= 15:
            result['confidence'] = 'Low'
        else:
            result['confidence'] = 'None'
            if not reasons:
                reasons.append("• No strong CGNAT indicators detected")
        
        result['reasons'] = reasons
        result['cgnat_score'] = cgnat_score
        
        return result
    
    def format_report(self, analysis: Dict) -> str:
        """Format analysis results as a readable report"""
        if analysis.get('error'):
            return f"ERROR: {analysis['error']}"
        
        report = []
        report.append("="*70)
        report.append("CGNAT DETECTION REPORT")
        report.append("="*70)
        report.append(f"\nTarget: {analysis['target']}")
        report.append(f"Resolved IP: {analysis['ip']}")
        report.append(f"Local IP: {analysis['local_ip']}")
        report.append("")
        
        # IP Classification
        report.append("IP Classification:")
        report.append(f"  Private IP (RFC 1918): {analysis['is_private']}")
        report.append(f"  CGNAT Range (100.64.0.0/10): {analysis['is_cgnat']}")
        report.append(f"  Loopback: {analysis['is_loopback']}")
        report.append("")
        
        # Traceroute
        if analysis['traceroute']:
            report.append("Traceroute Hops:")
            for ttl, hop in analysis['traceroute']:
                hop_str = hop if hop else "* * *"
                report.append(f"  {ttl:2d}: {hop_str}")
            report.append("")
        elif not self.has_raw_socket_permission:
            report.append("Traceroute: Skipped (requires elevated privileges)")
            report.append("")
        
        # Port Results
        if analysis['port_results']:
            report.append("TCP Port Check:")
            for port, is_open in analysis['port_results']:
                status = "OPEN" if is_open else "CLOSED/FILTERED"
                report.append(f"  Port {port}: {status}")
            report.append("")
        
        # Analysis
        report.append("CGNAT Analysis:")
        report.append(f"  Detection Confidence: {analysis['confidence']}")
        report.append(f"  CGNAT Detected: {analysis['cgnat_detected']}")
        report.append(f"  Score: {analysis.get('cgnat_score', 0)}/100")
        report.append("")
        
        report.append("Indicators:")
        for reason in analysis['reasons']:
            report.append(f"  {reason}")
        
        report.append("="*70)
        
        return "\n".join(report)


def test_cgnat_detector():
    """Test the CGNAT detector"""
    print("Testing CGNAT Detector...")
    print()
    
    detector = CGNATDetector()
    
    # Test cases
    test_targets = [
        "8.8.8.8",          # Google DNS (public)
        "192.168.1.1",      # Private IP
        "100.64.0.1",       # CGNAT range
        "127.0.0.1",        # Loopback
    ]
    
    for target in test_targets:
        print(f"\nTesting: {target}")
        print("-" * 70)
        
        analysis = detector.analyze_cgnat(target, skip_traceroute=True)
        report = detector.format_report(analysis)
        print(report)
        print()


if __name__ == "__main__":
    test_cgnat_detector()
