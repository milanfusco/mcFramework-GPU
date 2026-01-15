## mcFramework-GPU

CPU (NumPy) vs GPU (PyTorch/CUDA) Monte Carlo pi estimation demos + benchmarks.

### Setup

- **Python**: 3.11+

Using `uv` (recommended):

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

- **Simple demo**:

```bash
python -m examples.simple_demo
```

- **Benchmark (writes JSON into `results/`)**:

```bash
python -m pi_estimation.benchmark
```

- **Plots (reads JSON and writes PNGs into `results/`)**:

```bash
python -m pi_estimation.visualize
```

### GPU / CUDA

If `torch.cuda.is_available()` is `False`, everything will still run in CPU-only mode.
To use CUDA, install a CUDA-enabled PyTorch build appropriate for your system (see PyTorch install instructions).

