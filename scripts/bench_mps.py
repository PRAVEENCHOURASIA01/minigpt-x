"""
Apple Silicon MPS vs CPU Matrix Multiplication Benchmark.

This script benchmarks 2048x2048 float32 matrix multiplications on CPU and Apple Silicon MPS:
- Prints machine information (Python, PyTorch, OS, CPU, RAM)
- Performs 3 warm-up runs to ensure kernels are compiled and loaded
- Synchronizes with torch.mps.synchronize() before and after timing runs
- Computes average execution time over 20 runs
- Calculates TFLOPS (2 * n^3 / time / 1e12)
"""

import platform
import subprocess
import sys
import time
import torch


def get_sysctl(name: str) -> str:
    try:
        res = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception as e:
        return f"Unknown ({e})"


def print_machine_info():
    print("=== Machine Information ===")
    print(f"Python Path:     {sys.executable}")
    print(f"Python Version:  {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Built:       {torch.backends.mps.is_built()}")
    print(f"MPS Available:   {torch.backends.mps.is_available()}")
    print(f"macOS Version:   {platform.mac_ver()[0] or platform.platform()}")
    print(f"Chip / CPU:      {get_sysctl('machdep.cpu.brand_string')}")
    
    memsize_raw = get_sysctl("hw.memsize")
    try:
        ram_gb = f"{int(memsize_raw) / (1024 ** 3):.2f} GB"
    except ValueError:
        ram_gb = memsize_raw
    print(f"Total RAM:       {ram_gb}")
    print("===========================\n")


def benchmark_matmul(device_str: str, size: int = 2048, warmups: int = 3, runs: int = 20):
    device = torch.device(device_str)
    print(f"--- Benchmarking {device_str.upper()} ({size}x{size} float32 matmul) ---")
    
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)

    # Note on Warm-up Runs:
    # -------------------
    # GPUs and Apple Silicon MPS compile computation graphs, allocate internal driver buffers,
    # and warm up hardware frequency scaling on the initial execution. Timing the initial runs
    # would incorrectly include compilation and allocation overhead rather than steady-state compute.
    for _ in range(warmups):
        c = torch.matmul(a, b)
    if device_str == "mps":
        torch.mps.synchronize()

    # Note on torch.mps.synchronize():
    # --------------------------------
    # MPS (Metal Performance Shaders) execution in PyTorch is asynchronous. When torch.matmul()
    # is called, the operation is queued to the GPU command buffer and Python immediately continues.
    # Without calling torch.mps.synchronize(), standard wall-clock timing measures only the time
    # taken to enqueue the operation, drastically under-measuring true GPU computation time.
    # Synchronizing immediately before timing starts ensures the queue is clean, and synchronizing
    # immediately after the timed loop ensures all GPU work has finished before recording the end time.
    if device_str == "mps":
        torch.mps.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(runs):
        c = torch.matmul(a, b)
    if device_str == "mps":
        torch.mps.synchronize()
    end_time = time.perf_counter()

    avg_time_s = (end_time - start_time) / runs
    avg_time_ms = avg_time_s * 1000.0

    # Total floating point operations for N x N matrix multiplication: 2 * N^3 operations
    flops = 2.0 * (size ** 3)
    tflops = (flops / avg_time_s) / 1e12

    print(f"Average time per matmul: {avg_time_ms:.2f} ms")
    print(f"Achieved Throughput:    {tflops:.2f} TFLOPS")
    print()
    return avg_time_ms, tflops


def main():
    print_machine_info()
    cpu_ms, cpu_tflops = benchmark_matmul("cpu")

    if torch.backends.mps.is_available():
        mps_ms, mps_tflops = benchmark_matmul("mps")
        speedup = cpu_ms / mps_ms if mps_ms > 0 else 0.0
        print(f"Summary: MPS is {speedup:.2f}x faster than CPU for {2048}x{2048} float32 matmul.")
    else:
        print("MPS backend is not available on this machine. Skipping MPS benchmark.")


if __name__ == "__main__":
    main()
