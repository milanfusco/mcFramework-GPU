"""GPU-accelerated Monte Carlo methods for estimating Pi.

This module demonstrates the same tensor operations as NumPy but with 
GPU acceleration via CUDA, showing dramatic speed improvements for large workloads.
"""

import time
from typing import Tuple 

import torch

def estimate_pi_pytorch_cpu(n_samples: int) -> Tuple[float, float]:
    """
    Pytorch CPU tensor implementation of Monte Carlo estimation of Pi.

    Similar to numpy_cpu but uses PyTorch tensors on CPU.
    Useful for comparing Pytorch overhead vs NumPy baseline.

    Args:
        n_samples: Number of random points to generate.

    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.
    """
    start = time.perf_counter()

    x = torch.rand(n_samples, device="cpu")
    y = torch.rand(n_samples, device="cpu")

    # vectorized distance calculation
    distance_squared = x * x + y * y
    inside = torch.sum(distance_squared <= 1.0)

    pi_estimate = 4 * inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start

def estimate_pi_pytorch_cuda(n_samples: int, device: str = "cuda") -> Tuple[float, float]:
    """
    Pytorch CUDA tensor implementation of Monte Carlo estimation of Pi.

    Uses PyTorch tensors on GPU.

    Args:
        n_samples: Number of random points to generate.
        device: CUDA device to use (default: "cuda" for first available GPU)
    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    
    start = time.perf_counter()

    # allocate tensors on GPU
    x = torch.rand(n_samples, device=device)
    y = torch.rand(n_samples, device=device)

    # vectorized distance calculation
    distance_squared = x * x + y * y

    # count inside points
    inside = torch.sum(distance_squared <= 1.0)
    pi_estimate = 4 * inside / n_samples

    end = time.perf_counter()
    return pi_estimate, end - start


def estimate_pi_pytorch_cuda_batched(
    n_samples: int,
    batch_size: int = 1_000_000,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Memory-efficient batched Pytorch CUDA tensor implementation of Monte Carlo estimation of Pi.

    Useful for extremely large workloads that don't fit in GPU memory.

    Args:
        n_samples: Total number of samples to generate.
        batch_size: Number of samples to process in each batch.
        device: CUDA device to use (default: "cuda" for first available GPU)
    Returns:
        Tuple[float, float]: Estimated Pi and time taken in seconds.

    Raises:
        RuntimeError: If CUDA is not available.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    
    start = time.perf_counter()

    inside = 0
    remaining = n_samples    

    while remaining > 0:
        current_batch = min(batch_size, remaining)
        x = torch.rand(current_batch, device=device)
        y = torch.rand(current_batch, device=device)
        inside += torch.sum(x * x + y * y <= 1.0).item()
        remaining -= current_batch

    pi_estimate = 4 * inside / n_samples
    end = time.perf_counter()
    return pi_estimate, end - start

def get_gpu_info() -> dict:
    """
    Get detailed information about the available CUDA GPUs.

    Returns:
        dict: Dictionary containing GPU information.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
        "current_device": torch.cuda.current_device(),
        "is_available": torch.cuda.is_available(),
        "is_bf16_supported": torch.cuda.is_bf16_supported(),
    }

if __name__=="__main__":
    print("GPU Information:")
    print(get_gpu_info())

    if not get_gpu_info()["available"]:
        print("No CUDA GPUs available. Please check your PyTorch installation.")
        exit(1)

    print("\nTesting PyTorch CUDA methods with 10M samples...")
    print()

    n = 10_000_000

    pi_est, t = estimate_pi_pytorch_cpu(n)
    print(f"PyTorch CPU: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds") 

    pi_est, t = estimate_pi_pytorch_cuda(n)
    print(f"PyTorch CUDA: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds") 

    pi_est, t = estimate_pi_pytorch_cuda_batched(n)
    print(f"PyTorch CUDA Batched: Pi estimate = {pi_est:.6f}, Time taken = {t:.4f} seconds") 