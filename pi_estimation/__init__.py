"""
mcFramework-GPU: GPU-Accelerated Monte Carlo Computing

A comparative study of CPU (NumPy) vs GPU (PyTorch) tensor operations
for Monte Carlo simulations.
"""

__version__ = "0.1.0"
__author__ = "Milan Fusco"

from .cpu_methods import estimate_pi_python, estimate_pi_numpy, estimate_pi_numpy_batched
from .gpu_methods import (
    estimate_pi_pytorch_cpu,
    estimate_pi_pytorch_gpu,
    estimate_pi_pytorch_cuda,
    estimate_pi_pytorch_cuda_batched,
    get_gpu_info
)

__all__ = [
    "estimate_pi_python",
    "estimate_pi_numpy",
    "estimate_pi_numpy_batched",
    "estimate_pi_pytorch_cpu",
    "estimate_pi_pytorch_gpu",
    "estimate_pi_pytorch_cuda",
    "estimate_pi_pytorch_cuda_batched",
    "get_gpu_info",
]
