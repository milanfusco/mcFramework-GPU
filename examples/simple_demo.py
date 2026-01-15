"""
Simple demonstration of CPU vs GPU tensor operations for Monte Carlo π estimation.

This is a quick-start example showing the basic usage of each method.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pi_estimation.cpu_methods import estimate_pi_numpy
from pi_estimation.gpu_methods import estimate_pi_pytorch_cpu, estimate_pi_pytorch_gpu, get_gpu_info
import torch


def main():
    print("=" * 70)
    print("Monte Carlo π Estimation: Simple Demo")
    print("=" * 70)
    print()
    
    # Check GPU availability
    gpu_info = get_gpu_info()
    print("System Information:")
    if gpu_info["available"]:
        print(f"  ✓ GPU Available: {gpu_info['device_name']}")
        print(f"  ✓ CUDA Version: {gpu_info['cuda_version']}")
    else:
        print(f"  ✗ GPU Not Available (CPU-only mode)")
    print()
    
    # Run demonstrations with increasing sample sizes
    sample_sizes = [100_000, 1_000_000, 10_000_000]
    
    for n in sample_sizes:
        print(f"{'─' * 70}")
        print(f"Sample Size: {n:,}")
        print(f"{'─' * 70}")
        
        # NumPy (CPU tensors)
        pi_est, time_cpu = estimate_pi_numpy(n)
        print(f"NumPy CPU:   π ≈ {pi_est:.7f}  (time: {time_cpu:.4f}s)")
        
        # PyTorch CPU
        pi_est, time_pt_cpu = estimate_pi_pytorch_cpu(n)
        print(f"PyTorch CPU: π ≈ {pi_est:.7f}  (time: {time_pt_cpu:.4f}s)")
        
        # PyTorch GPU (if available)
        if gpu_info["available"]:
            pi_est, time_gpu = estimate_pi_pytorch_gpu(n)
            speedup = time_cpu / time_gpu
            print(f"PyTorch GPU: π ≈ {pi_est:.7f}  (time: {time_gpu:.4f}s)")
            print(f"  → GPU Speedup: {speedup:.1f}x faster than NumPy CPU")
        
        print()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    
    if not gpu_info["available"]:
        print("GPU not available. Please check your PyTorch installation.")
        exit(1)

if __name__ == "__main__":
    main()
