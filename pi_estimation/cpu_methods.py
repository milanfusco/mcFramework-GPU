"""
CPU-based Monte Carlo methods for estimating Pi.

This module demonstrates tensor operations using pure Python and NumPy,
serving as a baseline for GPU-accelerated comparison.
"""

import random
import time 
from typing import Tuple

import numpy as np


def estimate_pi_python(n_samples: int) -> Tuple[float, float]:
    """
    Pure python implementation of Monte Carlo estimation of Pi.

    Uses scalar operations -slowest but most portable.

    Args:
        n_samples (int): Number of random points to generate.
    
    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.
    """
    start = time.perf_counter()

    inside = 0
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside += 1

    pi_estimate = 4 * inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start


def estimate_pi_numpy(n_samples: int) -> Tuple[float, float]:
    """
    NumPy tensor-based Monte Carlo estimation of Pi.

    Uses vectorized operations on CPU tensors (ndarrays)
    This is the approach used in the original mcFramework. 

    Args:
        n_samples (int): Number of random points to generate.
    
    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.
    """
    start = time.perf_counter()

    x=np.random.random(n_samples)
    y=np.random.random(n_samples)
    
    # vectorized distance calculation
    distance_squared = x * x + y * y
    inside = np.sum(distance_squared <= 1.0)

    pi_estimate = 4 * inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start


def estimate_pi_numpy_batched(n_samples: int, batch_size: int = 1_000_000) -> Tuple[float, float]:
    """
    Memory-efficient batched NumPy implementation of Monte Carlo estimation of Pi.

    Processes samples in batches to avoid memory issues with very large n_samples.

    Args:
        n_samples (int): Total number of samples to generate.
        batch_size (int): Number of samples to process in each batch.
    
    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.
    """
    start = time.perf_counter()

    inside = 0
    remaining = n_samples

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        x = np.random.random(current_batch)
        y = np.random.random(current_batch)
        inside += np.sum(x * x + y * y <= 1)
        remaining -= current_batch

    pi_estimate = 4 * inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start


if __name__=="__main__":
    # Quick test
    print("Testing CPU methods with 1M samples...")
    print()

    n = 1_000_000

    pi_est, t = estimate_pi_python(n)
    print(f"Pure Python: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds")

    pi_est, t = estimate_pi_numpy(n)
    print(f"NumPy: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds")

    pi_est, t = estimate_pi_numpy_batched(n)
    print(f"NumPy Batched: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds")