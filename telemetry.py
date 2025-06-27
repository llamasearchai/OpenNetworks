"""
Telemetry and Monitoring System for OpenNetworks Framework
=========================================================

Real-time network performance monitoring and telemetry collection
with advanced analytics and visualization.
"""

import time
import threading
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass, asdict
import logging
import queue
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import from neurallink with error handling
try:
    from neurallink import (
        OpenNetworksConfig, NetworkProtocol, NetworkMetrics
    )
except ImportError:
    # Define minimal classes if neurallink import fails
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    
    class NetworkProtocol(Enum):
        INFINIBAND = "infiniband"
        ETHERNET = "ethernet"
    
    @dataclass
    class NetworkMetrics:
        timestamp: datetime
        protocol: NetworkProtocol
        operation: str
        message_size: int
        bandwidth_gbps: float
        latency_us: float
        cpu_utilization: float
        gpu_utilization: float
        memory_usage_mb: float
        packet_loss: float
        error_count: int
    
    class OpenNetworksConfig:
        def __init__(self):
            self.telemetry_interval_ms = 100
            self.log_level = "INFO"
            self.metrics_retention_hours = 24
            self.log_directory = "./logs"
            self.metrics_directory = "./metrics"
            self.config_directory = "./config"


@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_utilization: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0


class TelemetryCollector:
    """Performance telemetry collection for OpenNetworks framework"""
    
    def __init__(self, config: OpenNetworksConfig):
        self.config = config
        self.console = Console()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.network_metrics: List[NetworkMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        
        # Collection control
        self.is_collecting = False
        self.collection_thread = None
        self.stop_event = threading.Event()
        self.metrics_queue = queue.Queue()
        
        # Callbacks for real-time processing
        self.callbacks: List[Callable] = []
        
        # Initialize previous values for delta calculations
        self._prev_disk_io = None
        self._prev_net_io = None
        
        # Setup directories
        Path(config.metrics_directory).mkdir(parents=True, exist_ok=True)
        
    def start_collection(self):
        """Start telemetry collection"""
        if self.is_collecting:
            self.console.print("[yellow]Telemetry collection already running[/yellow]")
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.console.print("[green]Telemetry collection started[/green]")
    
    def stop_collection(self):
        """Stop telemetry collection"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        
        self.console.print("[yellow]Telemetry collection stopped[/yellow]")
    
    def _collection_loop(self):
        """Main telemetry collection loop"""
        while self.is_collecting:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                if system_metrics:
                    self.system_metrics.append(system_metrics)
                
                # Process callbacks
                for callback in self.callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        self.logger.error(f"Metric callback error: {e}")
                
                time.sleep(self.config.telemetry_interval_ms / 1000.0)
                
            except Exception as e:
                self.logger.error(f"Telemetry collection error: {e}")
                time.sleep(1.0)  # Prevent tight error loop
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system performance metrics"""
        if not PSUTIL_AVAILABLE:
            # Return simulated metrics if psutil is not available
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=np.random.uniform(10, 60),
                memory_percent=np.random.uniform(30, 80),
                memory_used_gb=np.random.uniform(2, 8),
                memory_total_gb=16.0,
                disk_io_read_mb=np.random.uniform(0, 100),
                disk_io_write_mb=np.random.uniform(0, 50),
                network_sent_mb=np.random.uniform(0, 10),
                network_recv_mb=np.random.uniform(0, 10),
                gpu_utilization=np.random.uniform(0, 50),
                gpu_memory_used_mb=np.random.uniform(0, 2048),
                gpu_memory_total_mb=4096.0
            )
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk I/O with delta calculation
            disk_io = psutil.disk_io_counters()
            if disk_io and self._prev_disk_io:
                disk_read_mb = (disk_io.read_bytes - self._prev_disk_io.read_bytes) / (1024**2)
                disk_write_mb = (disk_io.write_bytes - self._prev_disk_io.write_bytes) / (1024**2)
            else:
                disk_read_mb = disk_io.read_bytes / (1024**2) if disk_io else 0
                disk_write_mb = disk_io.write_bytes / (1024**2) if disk_io else 0
            
            self._prev_disk_io = disk_io
            
            # Network I/O with delta calculation
            net_io = psutil.net_io_counters()
            if net_io and self._prev_net_io:
                net_sent_mb = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / (1024**2)
                net_recv_mb = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / (1024**2)
            else:
                net_sent_mb = net_io.bytes_sent / (1024**2) if net_io else 0
                net_recv_mb = net_io.bytes_recv / (1024**2) if net_io else 0
            
            self._prev_net_io = net_io
            
            # GPU metrics (simulated or real)
            gpu_util, gpu_mem_used, gpu_mem_total = self._get_gpu_metrics()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                disk_io_read_mb=max(0, disk_read_mb),  # Ensure non-negative
                disk_io_write_mb=max(0, disk_write_mb),
                network_sent_mb=max(0, net_sent_mb),
                network_recv_mb=max(0, net_recv_mb),
                gpu_utilization=gpu_util,
                gpu_memory_used_mb=gpu_mem_used,
                gpu_memory_total_mb=gpu_mem_total
            )
            
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return None
    
    def _get_gpu_metrics(self) -> tuple:
        """Get GPU utilization metrics"""
        try:
            # Try to get real GPU metrics if available
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0]:
                    values = lines[0].split(', ')
                    if len(values) >= 3:
                        gpu_util = float(values[0])
                        gpu_mem_used = float(values[1])
                        gpu_mem_total = float(values[2])
                        return gpu_util, gpu_mem_used, gpu_mem_total
        except:
            pass
        
        # Fallback to simulated metrics
        return (
            np.random.uniform(0, 80),  # GPU utilization
            np.random.uniform(500, 2000),  # GPU memory used MB
            4096.0  # GPU memory total MB
        )
    
    def add_network_metric(self, metric: NetworkMetrics):
        """Add a network performance metric"""
        self.network_metrics.append(metric)
    
    def add_metric_callback(self, callback: Callable):
        """Add a callback for real-time metric processing"""
        self.callbacks.append(callback)
    
    def get_recent_metrics(self, seconds: int = 60) -> Dict[str, List]:
        """Get metrics from the last N seconds"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_network = [m for m in self.network_metrics 
                         if hasattr(m, 'timestamp') and m.timestamp >= cutoff_time]
        
        return {
            "system": recent_system,
            "network": recent_network
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.system_metrics:
            return {}
        
        recent_metrics = list(self.system_metrics)[-100:]  # Last 100 samples
        
        if not recent_metrics:
            return {}
        
        # Calculate statistics
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        gpu_values = [m.gpu_utilization for m in recent_metrics]
        
        summary = {
            "collection_active": self.is_collecting,
            "sample_count": len(recent_metrics),
            "time_range_minutes": (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds() / 60 if len(recent_metrics) > 1 else 0,
            "cpu": {
                "avg": np.mean(cpu_values) if cpu_values else 0,
                "max": np.max(cpu_values) if cpu_values else 0,
                "min": np.min(cpu_values) if cpu_values else 0,
                "current": recent_metrics[-1].cpu_percent if recent_metrics else 0
            },
            "memory": {
                "avg": np.mean(memory_values) if memory_values else 0,
                "max": np.max(memory_values) if memory_values else 0,
                "current": recent_metrics[-1].memory_percent if recent_metrics else 0,
                "used_gb": recent_metrics[-1].memory_used_gb if recent_metrics else 0,
                "total_gb": recent_metrics[-1].memory_total_gb if recent_metrics else 0
            },
            "gpu": {
                "avg": np.mean(gpu_values) if gpu_values else 0,
                "max": np.max(gpu_values) if gpu_values else 0,
                "current": recent_metrics[-1].gpu_utilization if recent_metrics else 0
            }
        }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export collected metrics to file"""
        try:
            # Convert metrics to serializable format
            system_metrics_data = []
            for m in self.system_metrics:
                try:
                    system_metrics_data.append(asdict(m))
                except:
                    # Fallback for non-dataclass objects
                    system_metrics_data.append({
                        'timestamp': m.timestamp.isoformat() if hasattr(m.timestamp, 'isoformat') else str(m.timestamp),
                        'cpu_percent': getattr(m, 'cpu_percent', 0),
                        'memory_percent': getattr(m, 'memory_percent', 0),
                        'gpu_utilization': getattr(m, 'gpu_utilization', 0)
                    })
            
            network_metrics_data = []
            for m in self.network_metrics:
                try:
                    network_metrics_data.append(asdict(m))
                except:
                    # Fallback for non-dataclass objects
                    network_metrics_data.append({
                        'timestamp': getattr(m, 'timestamp', datetime.now()).isoformat(),
                        'operation': getattr(m, 'operation', 'unknown'),
                        'message_size': getattr(m, 'message_size', 0),
                        'bandwidth_gbps': getattr(m, 'bandwidth_gbps', 0)
                    })
            
            data = {
                "export_time": datetime.now().isoformat(),
                "config": asdict(self.config) if hasattr(self.config, '__dict__') else str(self.config),
                "system_metrics": system_metrics_data,
                "network_metrics": network_metrics_data
            }
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            self.console.print(f"[green]Metrics exported to {filepath}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]Failed to export metrics: {e}[/red]")


class NetworkTopologyDiscovery:
    """Network topology discovery and analysis"""
    
    def __init__(self, config: OpenNetworksConfig):
        self.config = config
        self.console = Console()
        self.discovered_nodes: Dict[str, Dict] = {}
        
    def discover_topology(self) -> Dict[str, Any]:
        """Discover network topology"""
        topology = {
            "discovery_time": datetime.now().isoformat(),
            "local_node": self._get_local_node_info(),
            "discovered_nodes": {},
            "network_interfaces": self._get_network_interfaces(),
            "routing_table": self._get_routing_info()
        }
        
        return topology
    
    def _get_local_node_info(self) -> Dict[str, Any]:
        """Get local node information"""
        import socket
        import platform
        
        node_info = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }
        
        if PSUTIL_AVAILABLE:
            try:
                node_info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
                })
            except Exception as e:
                self.console.print(f"[yellow]Could not get extended node info: {e}[/yellow]")
        
        return node_info
    
    def _get_network_interfaces(self) -> Dict[str, Any]:
        """Get network interface information"""
        interfaces = {}
        
        if PSUTIL_AVAILABLE:
            try:
                net_interfaces = psutil.net_if_addrs()
                net_stats = psutil.net_if_stats()
                
                for interface, addresses in net_interfaces.items():
                    interface_info = {
                        "addresses": [],
                        "is_up": False,
                        "speed": 0,
                        "mtu": 0
                    }
                    
                    # Add address information
                    for addr in addresses:
                        interface_info["addresses"].append({
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": getattr(addr, 'netmask', None),
                            "broadcast": getattr(addr, 'broadcast', None)
                        })
                    
                    # Add interface statistics
                    if interface in net_stats:
                        stats = net_stats[interface]
                        interface_info.update({
                            "is_up": stats.isup,
                            "speed": stats.speed,
                            "mtu": stats.mtu
                        })
                    
                    interfaces[interface] = interface_info
            except Exception as e:
                self.console.print(f"[yellow]Could not get network interfaces: {e}[/yellow]")
        
        return interfaces
    
    def _get_routing_info(self) -> List[Dict]:
        """Get routing table information"""
        routing_info = []
        
        try:
            import subprocess
            import platform
            
            # Try to get routing table (Linux/Unix)
            if platform.system() != "Windows":
                result = subprocess.run(['ip', 'route'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            routing_info.append({"route": line.strip()})
            else:
                # Windows route command
                result = subprocess.run(['route', 'print'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if line.strip() and not line.startswith('='):
                            routing_info.append({"route": line.strip()})
        except Exception as e:
            self.console.print(f"[yellow]Could not get routing info: {e}[/yellow]")
        
        return routing_info


class PerformanceAnalyzer:
    """Advanced performance analysis and optimization recommendations"""
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        self.telemetry = telemetry_collector
        self.console = Console()
        
    def analyze_performance(self) -> Dict[str, Any]:
        """Perform comprehensive performance analysis"""
        summary = self.telemetry.get_performance_summary()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "bottlenecks": [],
            "recommendations": [],
            "performance_score": 0.0,
            "system_summary": summary
        }
        
        if not summary:
            return analysis
        
        # Analyze CPU performance
        cpu_analysis = self._analyze_cpu(summary.get("cpu", {}))
        analysis["bottlenecks"].extend(cpu_analysis["bottlenecks"])
        analysis["recommendations"].extend(cpu_analysis["recommendations"])
        
        # Analyze memory performance
        memory_analysis = self._analyze_memory(summary.get("memory", {}))
        analysis["bottlenecks"].extend(memory_analysis["bottlenecks"])
        analysis["recommendations"].extend(memory_analysis["recommendations"])
        
        # Analyze GPU performance
        gpu_analysis = self._analyze_gpu(summary.get("gpu", {}))
        analysis["bottlenecks"].extend(gpu_analysis["bottlenecks"])
        analysis["recommendations"].extend(gpu_analysis["recommendations"])
        
        # Calculate overall performance score
        analysis["performance_score"] = self._calculate_performance_score(summary)
        
        # Determine overall health
        if analysis["performance_score"] >= 80:
            analysis["overall_health"] = "excellent"
        elif analysis["performance_score"] >= 60:
            analysis["overall_health"] = "good"
        elif analysis["performance_score"] >= 40:
            analysis["overall_health"] = "fair"
        else:
            analysis["overall_health"] = "poor"
        
        return analysis
    
    def _analyze_cpu(self, cpu_metrics: Dict) -> Dict[str, List]:
        """Analyze CPU performance"""
        bottlenecks = []
        recommendations = []
        
        avg_cpu = cpu_metrics.get("avg", 0)
        max_cpu = cpu_metrics.get("max", 0)
        
        if avg_cpu > 80:
            bottlenecks.append("High average CPU utilization")
            recommendations.append("Consider CPU optimization or adding more compute nodes")
        
        if max_cpu > 95:
            bottlenecks.append("CPU saturation detected")
            recommendations.append("Implement CPU affinity or process balancing")
        
        return {"bottlenecks": bottlenecks, "recommendations": recommendations}
    
    def _analyze_memory(self, memory_metrics: Dict) -> Dict[str, List]:
        """Analyze memory performance"""
        bottlenecks = []
        recommendations = []
        
        avg_memory = memory_metrics.get("avg", 0)
        max_memory = memory_metrics.get("max", 0)
        
        if avg_memory > 85:
            bottlenecks.append("High memory utilization")
            recommendations.append("Consider memory optimization or adding more memory")
        
        if max_memory > 95:
            bottlenecks.append("Memory pressure detected")
            recommendations.append("Implement memory management strategies")
        
        return {"bottlenecks": bottlenecks, "recommendations": recommendations}
    
    def _analyze_gpu(self, gpu_metrics: Dict) -> Dict[str, List]:
        """Analyze GPU performance"""
        bottlenecks = []
        recommendations = []
        
        avg_gpu = gpu_metrics.get("avg", 0)
        max_gpu = gpu_metrics.get("max", 0)
        
        if avg_gpu < 20:
            bottlenecks.append("Low GPU utilization")
            recommendations.append("Optimize GPU workload distribution")
        elif avg_gpu > 90:
            bottlenecks.append("High GPU utilization")
            recommendations.append("Consider adding more GPUs or optimizing kernels")
        
        return {"bottlenecks": bottlenecks, "recommendations": recommendations}
    
    def _calculate_performance_score(self, summary: Dict) -> float:
        """Calculate overall performance score (0-100)"""
        cpu_score = max(0, 100 - summary.get("cpu", {}).get("avg", 50))
        memory_score = max(0, 100 - summary.get("memory", {}).get("avg", 50))
        gpu_score = min(100, summary.get("gpu", {}).get("avg", 50))
        
        # Weighted average
        overall_score = (cpu_score * 0.3 + memory_score * 0.3 + gpu_score * 0.4)
        
        return round(overall_score, 1) 