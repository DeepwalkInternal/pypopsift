#!/usr/bin/env python3
"""
Comprehensive test runner for pypopsift module.
"""

import sys
import subprocess

def main():

    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/", 
        "-v"
    ])
    

if __name__ == "__main__":
    main() 