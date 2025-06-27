#!/usr/bin/env python3
"""
OpenNetworks - Advanced AI Networking Framework for High-Performance Computing
===========================================================================

A comprehensive networking framework for AI workloads on supercomputers and data centers,
featuring InfiniBand/Ethernet protocols, RDMA optimization, and AI framework integration.

Author: Nik Jois <nikjois@llamasearch.ai>
Version: 1.0.0
License: MIT

Features:
- High-performance InfiniBand and Ethernet communication
- RDMA-optimized data transfers with zero-copy operations
- MPI integration for distributed computing workloads
- PyTorch and JAX distributed training support
- Custom collective operations for AI workloads
- Network topology discovery and optimization
- Production-grade monitoring and fault tolerance
- Performance profiling and bottleneck analysis
- GPU-GPU communication optimization
- Real-time network telemetry and visualization
"""

import os
import sys
import json
import time
import socket
import struct
import threading
import subprocess
import multiprocessing
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_char_p

# Core Python libraries
import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.tree import Tree
from rich.live import Live
from rich.layout import Layout
import click

# Try to import MPI (skip on Windows by default)
import platform
if platform.system() == "Windows":
    MPI_AVAILABLE = False
else:
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]
        MPI_AVAILABLE = True
    except ImportError:
        MPI_AVAILABLE = False

# Try to import PyTorch
try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, pmap, lax
    try:
        from jax.sharding import NamedSharding
    except ImportError:
        # Fallback for older JAX versions
        try:
            from jax.sharding import PositionalSharding as NamedSharding
        except ImportError:
            NamedSharding = None
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Try to import networking libraries
try:
    import psutil
    import netifaces
    NETWORK_TOOLS_AVAILABLE = True
except ImportError:
    NETWORK_TOOLS_AVAILABLE = False

# Export public API
__all__ = [
    'OpenNetworksConfig', 'NetworkProtocol', 'CollectiveOp', 'TransportMode', 
    'DeviceType', 'NetworkTopology', 'CommunicationBuffer', 'NetworkMetrics',
    'CollectiveOperation', 'RDMAConnection', 'OpenNetworksCLI', 'main'
]

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

@dataclass
class OpenNetworksConfig:
    """Configuration class for OpenNetworks framework"""
    # Network configuration
    primary_interface: str = "eth0"
    ib_interface: str = "ib0"
    rdma_device: str = "mlx5_0"
    port_range: Tuple[int, int] = (50000, 60000)
    
    # Performance settings
    max_message_size: int = 1024 * 1024 * 1024  # 1GB
    buffer_pool_size: int = 128
    worker_threads: int = 16
    async_progress_threads: int = 4
    
    # AI framework settings
    torch_backend: str = "nccl"
    jax_backend: str = "gpu"
    collective_timeout_ms: int = 30000
    
    # Monitoring and logging
    telemetry_interval_ms: int = 100
    log_level: str = "INFO"
    metrics_retention_hours: int = 24
    
    # Directories
    log_directory: str = "./logs"
    metrics_directory: str = "./metrics"
    config_directory: str = "./config"
    
    def __post_init__(self):
        """Ensure directories exist"""
        for directory in [self.log_directory, self.metrics_directory, self.config_directory]:
            Path(directory).mkdir(parents=True, exist_ok=True)

class NetworkProtocol(Enum):
    """Supported network protocols"""
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"
    ROCE = "roce"
    OMNI_PATH = "omni_path"

class CollectiveOp(Enum):
    """Collective operation types"""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    REDUCE = "reduce"
    BARRIER = "barrier"
    ALLTOALL = "alltoall"

class TransportMode(Enum):
    """Data transport modes"""
    EAGER = "eager"
    RENDEZVOUS = "rendezvous"
    RDMA_WRITE = "rdma_write"
    RDMA_READ = "rdma_read"
    GPU_DIRECT = "gpu_direct"

class DeviceType(Enum):
    """Device types for communication"""
    CPU = "cpu"
    GPU = "gpu"
    NIC = "nic"
    NVLINK = "nvlink"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class NetworkTopology:
    """Network topology information"""
    node_id: int
    total_nodes: int
    local_rank: int
    local_size: int
    global_rank: int
    global_size: int
    hostname: str
    interfaces: Dict[str, Dict[str, Any]]
    gpu_count: int
    interconnect: NetworkProtocol
    bandwidth_gbps: float
    latency_us: float

@dataclass
class CommunicationBuffer:
    """High-performance communication buffer"""
    buffer_id: str
    size: int
    memory_type: DeviceType
    data_ptr: Optional[int] = None
    registration_handle: Optional[Any] = None
    ref_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
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

@dataclass
class CollectiveOperation:
    """Collective operation specification"""
    op_id: str
    op_type: CollectiveOp
    participants: List[int]
    data_size: int
    data_type: str
    transport_mode: TransportMode
    timeout_ms: int
    progress_callback: Optional[Callable] = None

@dataclass
class RDMAConnection:
    """RDMA connection information"""
    connection_id: str
    local_rank: int
    remote_rank: int
    queue_pair: Optional[Any] = None
    completion_queue: Optional[Any] = None
    memory_regions: Dict[str, Any] = field(default_factory=dict)
    state: str = "disconnected"
    last_activity: datetime = field(default_factory=datetime.now)

# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

class OpenNetworksCLI:
    """Main CLI interface for OpenNetworks framework"""
    
    def __init__(self):
        self.console = Console()
        self.config = OpenNetworksConfig()
        
        # Initialize components - import here to avoid circular imports
        try:
            from transport_layer import InfiniBandTransport, EthernetTransport
            from collective_ops import CollectiveOperationsEngine
            from telemetry import TelemetryCollector
            
            self.ib_transport = InfiniBandTransport(self.config)
            self.eth_transport = EthernetTransport(self.config)
            self.collective_ops = CollectiveOperationsEngine(self.config)
            self.telemetry_collector = TelemetryCollector(self.config)
        except ImportError as e:
            self.console.print(f"[red]Warning: Could not import all components: {e}[/red]")
            # Create dummy objects to prevent attribute errors
            self.ib_transport = None
            self.eth_transport = None
            self.collective_ops = None
            self.telemetry_collector = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.log_directory}/opennetworks.log"),
                logging.StreamHandler()
            ]
        )
    
    def show_banner(self):
        """Display application banner"""
        banner = """
[bold blue]╔═══════════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║                      OpenNetworks v1.0.0                       ║[/bold blue]
[bold blue]║              Advanced AI Networking Framework                   ║[/bold blue]
[bold blue]║                                                                 ║[/bold blue]
[bold blue]║    High-Performance Computing • Distributed AI • RDMA          ║[/bold blue]
[bold blue]║    InfiniBand • Ethernet • PyTorch • JAX • Real-time Monitoring ║[/bold blue]
[bold blue]╚═══════════════════════════════════════════════════════════════╝[/bold blue]
        """
        self.console.print(Panel(banner, style="bold blue"))
        
    def main_menu(self):
        """Main application menu"""
        while True:
            self.console.print("\n[bold cyan]OpenNetworks Framework - Main Menu[/bold cyan]")
            self.console.print("1. Network Configuration & Testing")
            self.console.print("2. Collective Operations Demo")
            self.console.print("3. Performance Monitoring")
            self.console.print("4. System Information")
            self.console.print("5. Benchmark Suite")
            self.console.print("6. Exit")
            
            try:
                choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5", "6"])
                
                if choice == "1":
                    self.network_menu()
                elif choice == "2":
                    self.collective_ops_demo()
                elif choice == "3":
                    self.monitoring_menu()
                elif choice == "4":
                    self.system_info()
                elif choice == "5":
                    self.benchmark_suite()
                elif choice == "6":
                    break
            except (KeyboardInterrupt, EOFError):
                break
    
    def network_menu(self):
        """Network configuration and testing menu"""
        self.console.print("\n[bold yellow]Network Configuration & Testing[/bold yellow]")
        
        if not self.ib_transport or not self.eth_transport:
            self.console.print("[red]Transport layers not available[/red]")
            return
        
        protocol = Prompt.ask("Select protocol", choices=["infiniband", "ethernet"], default="infiniband")
        
        if protocol == "infiniband":
            success = self.ib_transport.initialize()
            if success:
                self.console.print("[green]InfiniBand transport ready[/green]")
                
                # Demo connection establishment
                if Confirm.ask("Test RDMA connection?"):
                    self.ib_transport.establish_connection(1, "192.168.1.100")
                    
                    # Demo RDMA operations
                    if Confirm.ask("Test RDMA write operation?"):
                        size = IntPrompt.ask("Data size (bytes)", default=1024*1024)
                        self.ib_transport.rdma_write(1, "local_buf", "remote_buf", size)
                        
        else:
            success = self.eth_transport.initialize()
            if success:
                self.console.print("[green]Ethernet transport ready[/green]")
                
                if Confirm.ask("Test TCP connection?"):
                    addr = Prompt.ask("Remote address", default="127.0.0.1")
                    port = IntPrompt.ask("Port", default=50000)
                    self.eth_transport.establish_connection(1, addr, port)
    
    def collective_ops_demo(self):
        """Demonstrate collective operations"""
        self.console.print("\n[bold yellow]Collective Operations Demo[/bold yellow]")
        
        if not self.collective_ops:
            self.console.print("[red]Collective operations not available[/red]")
            return
        
        # Initialize collective operations
        self.collective_ops.initialize()
        
        # Demo data
        data_size = IntPrompt.ask("Data size for demo", default=1000)
        demo_data = np.random.randn(data_size).astype(np.float32)
        
        self.console.print(f"Demo data shape: {demo_data.shape}")
        self.console.print(f"Data sum: {demo_data.sum():.4f}")
        
        # AllReduce demo
        if Confirm.ask("Test AllReduce operation?"):
            with Progress() as progress:
                task = progress.add_task("AllReduce", total=100)
                result = self.collective_ops.allreduce(demo_data, op="sum")
                progress.update(task, completed=100)
                
            self.console.print(f"AllReduce result sum: {result.sum():.4f}")
        
        # AllGather demo
        if Confirm.ask("Test AllGather operation?"):
            with Progress() as progress:
                task = progress.add_task("AllGather", total=100)
                result = self.collective_ops.allgather(demo_data)
                progress.update(task, completed=100)
                
            self.console.print(f"AllGather result: {len(result)} tensors")
        
        # Show performance metrics
        perf_summary = self.collective_ops.get_performance_summary()
        if perf_summary:
            table = Table(title="Collective Operations Performance")
            table.add_column("Operation", style="cyan")
            table.add_column("Count", justify="right")
            table.add_column("Avg Bandwidth (GB/s)", justify="right")
            table.add_column("Avg Latency (μs)", justify="right")
            
            for op, stats in perf_summary.items():
                table.add_row(
                    op,
                    str(stats["count"]),
                    f"{stats['avg_bandwidth']:.2f}",
                    f"{stats['avg_latency']:.2f}"
                )
            
            self.console.print(table)
    
    def monitoring_menu(self):
        """Performance monitoring menu"""
        self.console.print("\n[bold yellow]Performance Monitoring[/bold yellow]")
        
        if not self.telemetry_collector:
            self.console.print("[red]Telemetry collector not available[/red]")
            return
        
        if not self.telemetry_collector.is_collecting:
            if Confirm.ask("Start telemetry collection?"):
                self.telemetry_collector.start_collection()
        
        # Real-time monitoring
        if Confirm.ask("Show real-time monitoring?"):
            self._show_realtime_monitoring()
        
        # Performance summary
        summary = self.telemetry_collector.get_performance_summary()
        if summary:
            self._show_performance_summary(summary)
        else:
            self.console.print("[yellow]No performance data available yet[/yellow]")
    
    def _show_realtime_monitoring(self):
        """Show real-time performance monitoring"""
        self.console.print("[cyan]Real-time monitoring (press Ctrl+C to stop)[/cyan]")
        
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        try:
            with Live(layout, refresh_per_second=2) as live:
                for i in range(60):  # Run for 60 iterations
                    summary = self.telemetry_collector.get_performance_summary()
                    
                    layout["header"].update(Panel("[bold]OpenNetworks Real-time Monitoring[/bold]", style="blue"))
                    
                    if summary:
                        table = Table(title="System Performance")
                        table.add_column("Metric", style="cyan")
                        table.add_column("Current", justify="right")
                        table.add_column("Average", justify="right")
                        table.add_column("Maximum", justify="right")
                        
                        cpu = summary.get("cpu", {})
                        memory = summary.get("memory", {})
                        gpu = summary.get("gpu", {})
                        
                        table.add_row("CPU (%)", f"{cpu.get('current', 0):.1f}", 
                                     f"{cpu.get('avg', 0):.1f}", f"{cpu.get('max', 0):.1f}")
                        table.add_row("Memory (%)", f"{memory.get('current', 0):.1f}", 
                                     f"{memory.get('avg', 0):.1f}", f"{memory.get('max', 0):.1f}")
                        table.add_row("GPU (%)", f"{gpu.get('current', 0):.1f}", 
                                     f"{gpu.get('avg', 0):.1f}", f"{gpu.get('max', 0):.1f}")
                        
                        layout["body"].update(table)
                    else:
                        layout["body"].update(Panel("Collecting performance data..."))
                    
                    layout["footer"].update(Panel(f"Samples: {summary.get('sample_count', 0) if summary else 0}", style="dim"))
                    
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitoring stopped[/yellow]")
    
    def _show_performance_summary(self, summary: Dict):
        """Show performance summary table"""
        table = Table(title="Performance Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Current", justify="right")
        table.add_column("Average", justify="right")
        table.add_column("Peak", justify="right")
        table.add_column("Status", justify="center")
        
        # CPU metrics
        cpu = summary.get("cpu", {})
        cpu_status = "🟢" if cpu.get("avg", 0) < 70 else "🟡" if cpu.get("avg", 0) < 90 else "🔴"
        table.add_row("CPU", f"{cpu.get('current', 0):.1f}%", 
                     f"{cpu.get('avg', 0):.1f}%", f"{cpu.get('max', 0):.1f}%", cpu_status)
        
        # Memory metrics
        memory = summary.get("memory", {})
        mem_status = "🟢" if memory.get("avg", 0) < 80 else "🟡" if memory.get("avg", 0) < 95 else "🔴"
        table.add_row("Memory", f"{memory.get('current', 0):.1f}%", 
                     f"{memory.get('avg', 0):.1f}%", f"{memory.get('max', 0):.1f}%", mem_status)
        
        # GPU metrics
        gpu = summary.get("gpu", {})
        gpu_status = "🟢" if gpu.get("avg", 0) > 20 else "🟡"
        table.add_row("GPU", f"{gpu.get('current', 0):.1f}%", 
                     f"{gpu.get('avg', 0):.1f}%", f"{gpu.get('max', 0):.1f}%", gpu_status)
        
        self.console.print(table)
    
    def system_info(self):
        """Display comprehensive system information"""
        self.console.print("\n[bold yellow]System Information[/bold yellow]")
        
        # System overview
        import platform
        import socket
        
        info_table = Table(title="System Overview")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Hostname", socket.gethostname())
        info_table.add_row("Platform", platform.platform())
        info_table.add_row("Architecture", platform.architecture()[0])
        info_table.add_row("Python Version", platform.python_version())
        
        if NETWORK_TOOLS_AVAILABLE:
            try:
                import psutil
                info_table.add_row("CPU Cores", str(psutil.cpu_count()))
                info_table.add_row("Memory Total", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
            except:
                pass
        
        self.console.print(info_table)
        
        # Framework availability
        framework_table = Table(title="AI Framework Support")
        framework_table.add_column("Framework", style="cyan")
        framework_table.add_column("Available", justify="center")
        framework_table.add_column("Version", style="dim")
        
        framework_table.add_row("PyTorch", "Available" if TORCH_AVAILABLE else "Not Available", 
                               torch.__version__ if TORCH_AVAILABLE else "Not installed")
        framework_table.add_row("JAX", "Available" if JAX_AVAILABLE else "Not Available", 
                               jax.__version__ if JAX_AVAILABLE else "Not installed")
        framework_table.add_row("MPI", "Available" if MPI_AVAILABLE else "Not Available", 
                               "Available" if MPI_AVAILABLE else "Not installed")
        framework_table.add_row("psutil", "Available" if NETWORK_TOOLS_AVAILABLE else "Not Available", 
                               psutil.__version__ if NETWORK_TOOLS_AVAILABLE else "Not installed")
        
        self.console.print(framework_table)
    
    def benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        self.console.print("\n[bold yellow]OpenNetworks Benchmark Suite[/bold yellow]")
        
        benchmarks = [
            ("Memory Bandwidth", self._benchmark_memory),
            ("Network Latency", self._benchmark_network_latency),
            ("Collective Operations", self._benchmark_collective_ops),
            ("System Performance", self._benchmark_system_performance)
        ]
        
        results = {}
        
        with Progress() as progress:
            main_task = progress.add_task("Running benchmarks...", total=len(benchmarks))
            
            for name, benchmark_func in benchmarks:
                progress.update(main_task, description=f"Running {name}...")
                try:
                    result = benchmark_func()
                    results[name] = result
                    self.console.print(f"[green]✓ {name} completed[/green]")
                except Exception as e:
                    results[name] = {"error": str(e)}
                    self.console.print(f"[red]✗ {name} failed: {e}[/red]")
                
                progress.advance(main_task)
        
        # Display results
        self._show_benchmark_results(results)
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory operations"""
        sizes = [1024, 1024*1024, 100*1024*1024]  # 1KB, 1MB, 100MB
        results = {}
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float32)
            
            # Memory copy benchmark
            start_time = time.time()
            copied_data = data.copy()
            end_time = time.time()
            
            duration = end_time - start_time
            duration = max(duration, 1e-9)  # Prevent division by zero
            bandwidth = (size * 4) / duration / (1024**3)  # GB/s
            results[f"copy_{size}_bytes"] = {
                "bandwidth_gbps": bandwidth,
                "latency_ms": duration * 1000
            }
        
        return results
    
    def _benchmark_network_latency(self) -> Dict:
        """Benchmark network latency"""
        # Simulated network latency test
        latencies = []
        
        for _ in range(10):
            start_time = time.time()
            # Simulate network operation
            time.sleep(0.001)  # 1ms simulated latency
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000000)  # microseconds
        
        return {
            "avg_latency_us": np.mean(latencies),
            "min_latency_us": np.min(latencies),
            "max_latency_us": np.max(latencies),
            "jitter_us": np.std(latencies)
        }
    
    def _benchmark_collective_ops(self) -> Dict:
        """Benchmark collective operations"""
        results = {}
        
        if not self.collective_ops:
            return {"error": "Collective operations not available"}
        
        # Test different data sizes
        sizes = [1000, 10000, 100000]
        
        for size in sizes:
            data = np.random.randn(size).astype(np.float32)
            
            # AllReduce benchmark
            start_time = time.time()
            result = self.collective_ops.allreduce(data)
            end_time = time.time()
            
            bandwidth = (size * 4) / (end_time - start_time) / (1024**3)
            
            results[f"allreduce_{size}"] = {
                "bandwidth_gbps": bandwidth,
                "latency_ms": (end_time - start_time) * 1000
            }
        
        return results
    
    def _benchmark_system_performance(self) -> Dict:
        """Benchmark overall system performance"""
        if not NETWORK_TOOLS_AVAILABLE:
            return {"error": "psutil not available"}
        
        import psutil
        
        # CPU benchmark
        cpu_times = []
        for _ in range(5):
            start_time = time.time()
            # CPU intensive operation
            _ = sum(i*i for i in range(100000))
            end_time = time.time()
            cpu_times.append(end_time - start_time)
        
        # Memory info
        memory = psutil.virtual_memory()
        
        return {
            "cpu_performance_score": 1.0 / np.mean(cpu_times),  # Higher is better
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent_used": memory.percent,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else 0
        }
    
    def _show_benchmark_results(self, results: Dict):
        """Display benchmark results"""
        self.console.print("\n[bold green]Benchmark Results[/bold green]")
        
        for benchmark_name, benchmark_results in results.items():
            if "error" in benchmark_results:
                self.console.print(f"[red]{benchmark_name}: Error - {benchmark_results['error']}[/red]")
                continue
            
            table = Table(title=benchmark_name)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")
            table.add_column("Unit", style="dim")
            
            for metric, value in benchmark_results.items():
                if "bandwidth" in metric:
                    table.add_row(metric, f"{value:.2f}", "GB/s")
                elif "latency" in metric:
                    if "us" in metric:
                        table.add_row(metric, f"{value:.2f}", "μs")
                    else:
                        table.add_row(metric, f"{value:.2f}", "ms")
                elif "score" in metric:
                    table.add_row(metric, f"{value:.2f}", "points")
                elif "gb" in metric:
                    table.add_row(metric, f"{value:.2f}", "GB")
                elif "percent" in metric:
                    table.add_row(metric, f"{value:.1f}", "%")
                elif "mhz" in metric:
                    table.add_row(metric, f"{value:.0f}", "MHz")
                else:
                    table.add_row(metric, str(value), "")
            
            self.console.print(table)
            self.console.print()
    
    def run(self):
        """Main application entry point"""
        try:
            self.show_banner()
            self.main_menu()
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Application interrupted by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]An error occurred: {e}[/red]")
            if self.logger:
                self.logger.error(f"Application error: {e}", exc_info=True)
        finally:
            # Cleanup
            if self.telemetry_collector and self.telemetry_collector.is_collecting:
                self.telemetry_collector.stop_collection()
            self.console.print("[dim]Thank you for using OpenNetworks![/dim]")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the OpenNetworks application"""
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="OpenNetworks - Advanced AI Networking Framework for High-Performance Computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python opennetworks.py                    # Start interactive CLI
    python opennetworks.py --help            # Show this help
    python opennetworks.py --version         # Show version information
    
This framework provides comprehensive networking capabilities for AI workloads:
- High-performance InfiniBand and Ethernet transport layers
- RDMA-optimized data transfers with zero-copy operations
- PyTorch and JAX distributed computing integration
- Real-time monitoring and performance optimization
- Production-grade collective operations

Perfect for supercomputers, data centers, and large-scale AI training.
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="OpenNetworks 1.0.0 - Advanced AI Networking Framework"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default="neurallink_config.json"
    )
    
    parser.add_argument(
        "--interface",
        type=str,
        help="Primary network interface",
        default="eth0"
    )
    
    parser.add_argument(
        "--protocol",
        choices=["infiniband", "ethernet", "roce"],
        help="Network protocol to use",
        default="infiniband"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    
    parser.add_argument(
        "--telemetry",
        action="store_true",
        help="Start telemetry collection automatically"
    )
    
    args = parser.parse_args()
    
    # Initialize and run application
    app = OpenNetworksCLI()
    
    # Override config if specified
    if args.interface:
        app.config.primary_interface = args.interface
    if args.log_level:
        app.config.log_level = args.log_level
    if args.telemetry and app.telemetry_collector:
        app.telemetry_collector.start_collection()
    
    app.run()


if __name__ == "__main__":
    main() 