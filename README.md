# LMC_AURIGA (Halo Benchmark)

This repo will hold three implementations (baseline, CPU-parallel, GPU) for “halo” benchmarks on Auriga data.
- All raw data (HDF5, Sun-position) must live under **data/** (git-ignored).
- Code goes in **src/**; driver scripts in **scripts/**; **results/** is for CSV outputs.
- Examples and final report notebooks live in **examples/** and **notebooks/**.
- A static HTML/PDF of the final report will go under **docs/**.

## Installation

This project supports **Python 3.10 or later**. The recommended approach is to
use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

Run the unit tests with:

```bash
pytest -q
```

The current test suite only covers the `anisotropy` helper but ensures the package can be imported.

The CUDA implementation significantly reduces runtime when computing speed distributions across the Auriga snapshots.

| Method | Avg Total Runtime (s) | Std Dev (s) | Avg per Snapshot (s) | Std Dev per Snapshot (s) |
|-------|----------------------|-------------|----------------------|--------------------------|
| Pandas/NumPy | 713.62 | 6.30 | 0.612 | 0.0054 |
| Numba CPU (parallel) | 399.49 | 2.15 | 0.343 | 0.0018 |
| CUDA (GPU) | 92.47 | 1.75 | 0.079 | 0.0015 |

See `examples/benchmark.ipynb` for a runnable demonstration of the pipeline.

## Next Steps

1. Refactor each notebook’s hard-coded halo IDs & snapshot lists into parameterized functions under src/.  
2. Ensure the three modules share a uniform API (snapshot(...) and 
un_all(...)).  
3. Write a single dispatcher script in scripts/ that chooses “baseline”, “cpu”, or “gpu” at runtime.
