"""
Visualization tools for benchmark results.

Creates publication-quality plots showing performance comparisons,
speedup analysis, and convergence behavior.
"""

import json
import math
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_results(filename: str = "benchmark_results.json") -> dict:
    """Load benchmark results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_performance_comparison(results: dict, output_file: str = "performance_comparison.png"):
    """
    Create log-log plot comparing execution times across methods and sample sizes.
    
    Args:
        results: Benchmark results dictionary
        output_file: Output filename for plot
    """
    # Organize data by method
    methods = {}
    for result in results["results"]:
        method = result["method"]
        if method not in methods:
            methods[method] = {"n_samples": [], "times": []}
        methods[method]["n_samples"].append(result["n_samples"])
        methods[method]["times"].append(result["time_seconds"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "Pure Python": "#e74c3c",
        "NumPy CPU": "#3498db",
        "PyTorch CPU": "#2ecc71",
        "PyTorch GPU": "#f39c12"
    }
    
    markers = {
        "Pure Python": "o",
        "NumPy CPU": "s",
        "PyTorch CPU": "^",
        "PyTorch GPU": "D"
    }
    
    for method, data in methods.items():
        ax.loglog(
            data["n_samples"], 
            data["times"],
            marker=markers.get(method, 'o'),
            linewidth=2,
            markersize=8,
            label=method,
            color=colors.get(method, None)
        )
    
    ax.set_xlabel("Number of Samples", fontsize=12, fontweight='bold')
    ax.set_ylabel("Execution Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title("Monte Carlo π Estimation: Performance Comparison\nCPU vs GPU Tensor Operations", 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved performance comparison to {output_file}")
    plt.close()


def plot_speedup_analysis(results: dict, output_file: str = "speedup_analysis.png"):
    """
    Create bar plot showing speedup factors relative to Pure Python baseline.
    
    Args:
        results: Benchmark results dictionary
        output_file: Output filename for plot
    """
    # Get largest sample size for comparison
    max_samples = max(r["n_samples"] for r in results["results"])
    
    # Get speedups for max sample size
    speedups = {}
    for result in results["results"]:
        if result["n_samples"] == max_samples:
            speedups[result["method"]] = result["speedup"]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(speedups.keys())
    speedup_values = list(speedups.values())
    
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    bars = ax.bar(methods, speedup_values, color=colors[:len(methods)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar, value in zip(bars, speedup_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel("Speedup Factor", fontsize=12, fontweight='bold')
    ax.set_title(f"Speedup Analysis (n={max_samples:,} samples)\nRelative to Pure Python Baseline",
                 fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved speedup analysis to {output_file}")
    plt.close()


def plot_convergence(results: dict, output_file: str = "convergence_analysis.png"):
    """
    Plot convergence to π as sample size increases.
    
    Args:
        results: Benchmark results dictionary
        output_file: Output filename for plot
    """
    # Organize data by method
    methods = {}
    for result in results["results"]:
        method = result["method"]
        if method not in methods:
            methods[method] = {"n_samples": [], "errors": []}
        methods[method]["n_samples"].append(result["n_samples"])
        methods[method]["errors"].append(result["error"])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "Pure Python": "#e74c3c",
        "NumPy CPU": "#3498db",
        "PyTorch CPU": "#2ecc71",
        "PyTorch GPU": "#f39c12"
    }
    
    for method, data in methods.items():
        ax.loglog(
            data["n_samples"], 
            data["errors"],
            marker='o',
            linewidth=2,
            markersize=8,
            label=method,
            color=colors.get(method, None),
            alpha=0.7
        )
    
    # Add theoretical convergence rate (1/sqrt(n))
    n_theory = np.logspace(3, 8, 100)
    error_theory = 1.0 / np.sqrt(n_theory)
    ax.loglog(n_theory, error_theory, 'k--', linewidth=2, label='Theoretical O(1/√n)', alpha=0.5)
    
    ax.set_xlabel("Number of Samples", fontsize=12, fontweight='bold')
    ax.set_ylabel("Absolute Error |estimate - π|", fontsize=12, fontweight='bold')
    ax.set_title("Convergence to π\nAll Methods Follow Expected O(1/√n) Behavior",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved convergence analysis to {output_file}")
    plt.close()


def plot_sample_visualization(n_samples: int = 10000, output_file: str = "sample_visualization.png"):
    """
    Create visual scatter plot showing Monte Carlo sampling for π estimation.
    
    Args:
        n_samples: Number of points to plot
        output_file: Output filename for plot
    """
    # Generate random points
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    
    # Determine which are inside circle
    inside = (x*x + y*y) <= 1.0
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot points
    ax.scatter(x[inside], y[inside], c='#2ecc71', s=1, alpha=0.5, label='Inside circle')
    ax.scatter(x[~inside], y[~inside], c='#e74c3c', s=1, alpha=0.5, label='Outside circle')
    
    # Draw quarter circle
    theta = np.linspace(0, np.pi/2, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Quarter circle')
    
    # Calculate π estimate
    pi_estimate = 4.0 * np.sum(inside) / n_samples
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Monte Carlo π Estimation Visualization\n'
                 f'n={n_samples:,} samples → π ≈ {pi_estimate:.6f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved sample visualization to {output_file}")
    plt.close()


def create_all_plots(results_file: str = "benchmark_results.json", output_dir: str = "results"):
    """
    Generate all visualization plots from benchmark results.
    
    Args:
        results_file: Path to benchmark results JSON
        output_dir: Directory to save plots
    """
    # Create output directory if needed
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load results
    results = load_results(results_file)
    
    print("Generating visualizations...")
    print()
    
    # Create all plots
    plot_performance_comparison(results, f"{output_dir}/performance_comparison.png")
    plot_speedup_analysis(results, f"{output_dir}/speedup_analysis.png")
    plot_convergence(results, f"{output_dir}/convergence_analysis.png")
    plot_sample_visualization(10000, f"{output_dir}/sample_visualization.png")
    
    print()
    print("All visualizations complete!")


if __name__ == "__main__":
    create_all_plots()
