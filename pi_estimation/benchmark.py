"""
Benchmarking framework for comparing CPU and GPU Monte Carlo methods.

Runs systematic performance comparisons across different sample sizes
and methods, collecting timing and accuracy data.
"""

import json
import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Tuple

import numpy as np
import torch

from .cpu_methods import estimate_pi_python, estimate_pi_numpy
from .gpu_methods import estimate_pi_pytorch_cpu, estimate_pi_pytorch_gpu, get_gpu_info


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    method: str
    n_samples: int
    pi_estimate: float
    time_seconds: float
    error: float  # |estimate - π|
    speedup: float = 1.0  # Relative to baseline
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Benchmarker:
    """
    Framework for running and collecting benchmark results.
    """
    
    def __init__(self, sample_sizes: List[int] = None):
        """
        Initialize benchmarker.
        
        Args:
            sample_sizes: List of sample sizes to test (default: powers of 10 from 1k to 100M)
        """
        if sample_sizes is None:
            sample_sizes = [10**i for i in range(3, 9)]  # 1k to 100M
        
        self.sample_sizes = sample_sizes
        self.results: List[BenchmarkResult] = []
        self.baseline_times: Dict[int, float] = {}
        
    def run_method(
        self, 
        name: str, 
        method: Callable[[int], Tuple[float, float]],
        n_samples: int
    ) -> BenchmarkResult:
        """
        Run a single benchmark.
        
        Args:
            name: Method name for display
            method: Function that takes n_samples and returns (pi_estimate, time)
            n_samples: Number of samples to use
            
        Returns:
            BenchmarkResult with timing and accuracy data
        """
        try:
            pi_estimate, time_seconds = method(n_samples)
            error = abs(pi_estimate - math.pi)
            
            # Calculate speedup relative to baseline (Pure Python)
            if name == "Pure Python":
                self.baseline_times[n_samples] = time_seconds
                speedup = 1.0
            else:
                baseline_time = self.baseline_times.get(n_samples, time_seconds)
                speedup = baseline_time / time_seconds if time_seconds > 0 else 0.0
            
            result = BenchmarkResult(
                method=name,
                n_samples=n_samples,
                pi_estimate=pi_estimate,
                time_seconds=time_seconds,
                error=error,
                speedup=speedup
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error running {name} with {n_samples} samples: {e}")
            return None
    
    def run_full_comparison(self, skip_python: bool = True, skip_large_python: bool = True):
        """
        Run complete benchmark comparing all methods across all sample sizes.
        
        Args:
            skip_python: Skip pure Python (it's very slow)
            skip_large_python: Skip pure Python for sample sizes > 1M
        """
        print("=" * 70)
        print("Monte Carlo π Estimation: CPU vs GPU Tensor Performance Comparison")
        print("=" * 70)
        print()
        
        # Display GPU info
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            print(f"GPU: {gpu_info['device_name']}")
            print(f"CUDA Version: {gpu_info['cuda_version']}")
        else:
            print("GPU: Not available (CPU-only benchmarks will run)")
        print()
        
        methods = [
            ("Pure Python", estimate_pi_python),
            ("NumPy CPU", estimate_pi_numpy),
            ("PyTorch CPU", estimate_pi_pytorch_cpu),
        ]
        
        if gpu_info["available"]:
            methods.append(("PyTorch GPU", estimate_pi_pytorch_gpu))
        
        for n_samples in self.sample_sizes:
            print(f"\n{'─' * 70}")
            print(f"Sample Size: {n_samples:,}")
            print(f"{'─' * 70}")
            
            for name, method in methods:
                # Skip pure Python for large samples (it's too slow)
                if name == "Pure Python":
                    if skip_python:
                        continue
                    if skip_large_python and n_samples > 1_000_000:
                        print(f"{name:15} Skipped (too slow for large n)")
                        continue
                
                result = self.run_method(name, method, n_samples)
                
                if result:
                    print(f"{name:15} "
                          f"π ≈ {result.pi_estimate:.7f}  "
                          f"error: {result.error:.2e}  "
                          f"time: {result.time_seconds:8.4f}s  "
                          f"speedup: {result.speedup:6.1f}x")
        
        print(f"\n{'=' * 70}")
        print("Benchmark Complete")
        print(f"{'=' * 70}\n")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """
        Save results to JSON file.
        
        Args:
            filename: Output filename
        """
        data = {
            "gpu_info": get_gpu_info(),
            "sample_sizes": self.sample_sizes,
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def get_summary_stats(self) -> dict:
        """
        Calculate summary statistics across all benchmarks.
        
        Returns:
            Dictionary with max speedups and average errors by method
        """
        methods = set(r.method for r in self.results)
        summary = {}
        
        for method in methods:
            method_results = [r for r in self.results if r.method == method]
            if method_results:
                summary[method] = {
                    "max_speedup": max(r.speedup for r in method_results),
                    "avg_error": np.mean([r.error for r in method_results]),
                    "min_time": min(r.time_seconds for r in method_results),
                    "max_time": max(r.time_seconds for r in method_results),
                }
        
        return summary


if __name__ == "__main__":
    # Run benchmark with reasonable defaults
    benchmarker = Benchmarker(
        sample_sizes=[10**i for i in range(4, 8)]  # 10k to 10M
    )
    
    benchmarker.run_full_comparison(skip_python=True)
    benchmarker.save_results("results/benchmark_results.json")
    
    print("\nSummary Statistics:")
    print("=" * 70)
    summary = benchmarker.get_summary_stats()
    for method, stats in summary.items():
        print(f"\n{method}:")
        print(f"  Max Speedup: {stats['max_speedup']:.1f}x")
        print(f"  Avg Error:   {stats['avg_error']:.2e}")
        print(f"  Time Range:  {stats['min_time']:.4f}s - {stats['max_time']:.4f}s")
