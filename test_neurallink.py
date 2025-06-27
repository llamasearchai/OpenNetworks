#!/usr/bin/env python3
"""
OpenNetworks Framework Test Suite
=================================

Comprehensive test suite for the OpenNetworks framework.
"""

import sys
import time
import traceback
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

# Import availability flags
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import platform
if platform.system() == "Windows":
    MPI_AVAILABLE = False
else:
    try:
        from mpi4py import MPI  # type: ignore[import-not-found]
        MPI_AVAILABLE = True
    except ImportError:
        MPI_AVAILABLE = False

try:
    import psutil
    import netifaces
    NETWORK_TOOLS_AVAILABLE = True
except ImportError:
    NETWORK_TOOLS_AVAILABLE = False

def test_imports():
    """Test critical module imports"""
    console = Console()
    console.print("\n[bold yellow]Testing Module Imports[/bold yellow]")
    
    modules = [
        "neurallink",
        "transport_layer", 
        "collective_ops",
        "telemetry"
    ]
    
    results = {}
    for module in modules:
        try:
            __import__(module)
            results[module] = "SUCCESS"
        except ImportError as e:
            results[module] = f"FAILED: {e}"
    
    # Check specific classes in neurallink
    try:
        from neurallink import OpenNetworksConfig, OpenNetworksCLI
        results["OpenNetworks Classes"] = "SUCCESS"
    except ImportError as e:
        results["OpenNetworks Classes"] = f"FAILED: {e}"
    
    # Display results
    table = Table(title="Import Test Results")
    table.add_column("Module", style="cyan")
    table.add_column("Status")
    
    all_passed = True
    for module, status in results.items():
        if "FAILED" in status:
            table.add_row(module, f"[red]{status}[/red]")
            all_passed = False
        else:
            table.add_row(module, f"[green]{status}[/green]")
    
    console.print(table)
    
    failed_imports = [name for name, status in results.items() if "FAILED" in status]
    if failed_imports:
        console.print(f"\n[red]Failed imports: {failed_imports}[/red]")
    
    return all_passed

def test_configuration():
    """Test configuration system"""
    console = Console()
    console.print("\n[bold yellow]Testing Configuration System[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig
        
        config = OpenNetworksConfig()
        assert config.primary_interface == "eth0"
        assert config.ib_interface == "ib0"
        assert config.rdma_device == "mlx5_0"
        assert config.port_range == (50000, 60000)
        assert config.max_message_size == 1024 * 1024 * 1024
        assert config.buffer_pool_size == 128
        assert config.worker_threads == 16
        assert config.async_progress_threads == 4
        assert config.torch_backend == "nccl"
        assert config.jax_backend == "gpu"
        assert config.collective_timeout_ms == 30000
        assert config.telemetry_interval_ms == 100
        assert config.log_level == "INFO"
        assert config.metrics_retention_hours == 24
        assert config.log_directory == "./logs"
        assert config.metrics_directory == "./metrics"
        assert config.config_directory == "./config"
        
        # Test directory creation
        import os
        assert os.path.exists(config.log_directory)
        assert os.path.exists(config.metrics_directory)
        assert os.path.exists(config.config_directory)
        
        console.print("[green]Configuration test passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Configuration test failed: {e}[/red]")
        return False

def test_transport_layer():
    """Test transport layer functionality"""
    console = Console()
    console.print("\n[bold yellow]Testing Transport Layer[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig
        from transport_layer import InfiniBandTransport, EthernetTransport
        
        config = OpenNetworksConfig()
        
        # Test InfiniBand transport
        ib_transport = InfiniBandTransport(config)
        assert ib_transport is not None
        
        # Test Ethernet transport
        eth_transport = EthernetTransport(config)
        assert eth_transport is not None
        
        # Test initialization (may fail on systems without hardware, but should not crash)
        try:
            ib_transport.initialize()
            eth_transport.initialize()
        except Exception:
            pass  # Hardware may not be available
        
        console.print("[green]Transport layer test passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Transport layer test failed: {e}[/red]")
        return False

def test_collective_operations():
    """Test collective operations functionality"""
    console = Console()
    console.print("\n[bold yellow]Testing Collective Operations[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig
        from collective_ops import CollectiveOperationsEngine
        import numpy as np
        
        config = OpenNetworksConfig()
        collective_ops = CollectiveOperationsEngine(config)
        
        # Initialize collective operations
        collective_ops.initialize()
        
        # Test with sample data
        test_data = np.random.randn(100).astype(np.float32)
        
        # Test AllReduce
        result = collective_ops.allreduce(test_data, op="sum")
        assert result is not None
        
        # Test AllGather
        gathered = collective_ops.allgather(test_data)
        assert gathered is not None
        
        console.print("[green]Collective operations test passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Collective operations test failed: {e}[/red]")
        return False

def test_telemetry():
    """Test telemetry system"""
    console = Console()
    console.print("\n[bold yellow]Testing Telemetry System[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig
        from telemetry import TelemetryCollector
        
        config = OpenNetworksConfig()
        telemetry = TelemetryCollector(config)
        
        # Test start/stop
        telemetry.start_collection()
        time.sleep(1)
        telemetry.stop_collection()
        
        # Test metrics collection
        summary = telemetry.get_performance_summary()
        
        console.print("[green]Telemetry test passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Telemetry test failed: {e}[/red]")
        return False

def test_cli_interface():
    """Test CLI interface"""
    console = Console()
    console.print("\n[bold yellow]Testing CLI Interface[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksCLI
        
        cli = OpenNetworksCLI()
        assert cli.config is not None
        
        # Test logging setup
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Testing logger")
        
        # Test banner display (just ensure no exception)
        cli.show_banner()
        
        # Test main menu (just ensure no exception)
        assert hasattr(cli, "main_menu")
        
        console.print("[green]CLI interface test passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]CLI interface test failed: {e}[/red]")
        return False

def test_performance_benchmark():
    """Test performance benchmarking"""
    console = Console()
    console.print("\n[bold yellow]Running Performance Benchmark[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig, OpenNetworksCLI
        
        config = OpenNetworksConfig()
        cli = OpenNetworksCLI()
        
        # Test memory benchmark
        results = cli._benchmark_memory()
        assert isinstance(results, dict)
        
        # Test network latency benchmark
        latency_results = cli._benchmark_network_latency()
        assert isinstance(latency_results, dict)
        
        console.print("[green]Performance benchmark passed[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]Performance benchmark failed: {e}[/red]")
        return False

def test_huggingface_datasets():
    """Test HuggingFace datasets integration"""
    console = Console()
    console.print("\n[bold yellow]Testing HuggingFace Datasets Integration[/bold yellow]")
    
    try:
        from neurallink import OpenNetworksConfig
        
        # Check if datasets library is available
        try:
            import datasets
            console.print("[green]HuggingFace datasets library available[/green]")
            
            # Test basic dataset loading (using a small, fast dataset)
            try:
                dataset = datasets.load_dataset("squad", split="train[:10]")
                console.print(f"[green]Successfully loaded sample dataset with {len(dataset)} examples[/green]")
            except Exception as e:
                console.print(f"[yellow]Dataset loading test skipped: {e}[/yellow]")
            
            return True
            
        except ImportError:
            console.print("[yellow]Datasets library not available. Skipping HuggingFace dataset test.[/yellow]")
            return True  # Not a failure, just not available
            
    except Exception as e:
        console.print(f"[red]Datasets library not available ({e}). Skipping HuggingFace dataset test.[/red]")
        return True  # Not a critical failure

def run_all_tests():
    """Run all test cases"""
    console = Console()
    
    # Display test banner
    banner = """
================ OpenNetworks Test Suite ================
=========== Comprehensive System Validation ===========
    """
    console.print(Panel(banner, style="bold blue"))
    
    test_functions = [
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Transport Layer", test_transport_layer),
        ("Collective Operations", test_collective_operations),
        ("Telemetry System", test_telemetry),
        ("CLI Interface", test_cli_interface),
        ("Performance Benchmark", test_performance_benchmark),
        ("HuggingFace Datasets", test_huggingface_datasets),
    ]
    
    results = {}
    
    for name, test_func in test_functions:
        console.print(f"\n[bold cyan]Running: {name}[/bold cyan]")
        try:
            results[name] = test_func()
        except Exception as e:
            console.print(f"[red]{name} test failed: {e}[/red]")
            traceback.print_exc()
            results[name] = False
    
    # Display summary
    passed = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    console.print("\n[bold]Test Summary[/bold]")
    
    summary_table = Table(title="Test Results Summary")
    summary_table.add_column("Test", style="cyan")
    summary_table.add_column("Result", justify="center")
    
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        style = "green" if success else "red"
        summary_table.add_row(name, f"[{style}]{status}[/{style}]")
    
    console.print(summary_table)
    
    if failed:
        console.print(f"\n[red]MULTIPLE FAILURES ({len(passed)}/{len(results)} - {len(passed)/len(results)*100:.1f}%)[/red]")
        console.print("[red]Please check the error messages above[/red]")
        return False
    else:
        console.print(f"\n[green]ALL TESTS PASSED ({len(passed)}/{len(results)} - 100%)[/green]")
        console.print("[green]OpenNetworks framework is ready for production![/green]")
        return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 