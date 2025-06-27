#!/usr/bin/env python3
"""
OpenNetworks Framework Entry Point
==================================

Alternative entry point for the OpenNetworks framework.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import sys
import os

# Add current directory to path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from neurallink import main
except ImportError as e:
    print(f"Error importing OpenNetworks: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

def main() -> None:  # pragma: no cover – thin wrapper
    """Entry point for OpenNetworks CLI"""
    from neurallink import main as neurallink_main
    neurallink_main()

if __name__ == "__main__":
    main() 