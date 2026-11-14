"""
Environment and hardware verification script for MiniGPT-X.
Checks Python version, PyTorch version, Apple Silicon MPS acceleration, macOS version, CPU brand, and total RAM.
"""

import platform
import subprocess
import sys
import torch


def get_sysctl(name: str) -> str:
    try:
        res = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"


def main():
    python_path = sys.executable
    python_version = platform.python_version()
    torch_version = torch.__version__
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    macos_version = platform.mac_ver()[0] or platform.platform()
    cpu_brand = get_sysctl("machdep.cpu.brand_string")
    
    memsize_raw = get_sysctl("hw.memsize")
    try:
        ram_gb = f"{int(memsize_raw) / (1024 ** 3):.2f} GB"
    except ValueError:
        ram_gb = memsize_raw

    print("=== MiniGPT-X Environment Check ===")
    print(f"Python Path:     {python_path}")
    print(f"Python Version:  {python_version}")
    print(f"PyTorch Version: {torch_version}")
    print(f"MPS Built:       {mps_built}")
    print(f"MPS Available:   {mps_available}")
    print(f"macOS Version:   {macos_version}")
    print(f"Chip / CPU:      {cpu_brand}")
    print(f"Total RAM:       {ram_gb}")
    print("===================================")

    if mps_built and mps_available:
        print("PASS: Apple Silicon MPS is usable.")
    else:
        print("FAIL: Apple Silicon MPS is NOT usable.")
        sys.exit(1)


if __name__ == "__main__":
    main()
