# LMC_AURIGA Benchmark Suite

This project contains three implementations for computing dark-matter speed distributions from the Auriga simulations. These mirror the versions used in Smith‑Orlik et al. (2023):

1. **Pandas/NumPy baseline** – straightforward CPU version
2. **Numba parallel CPU** – uses `@njit(parallel=True)`
3. **CUDA GPU** – accelerates the heavy computations on an NVIDIA GPU

Raw simulation data are **not** included. Place all HDF5 snapshots and Sun-position tables under the `data/` directory (which is ignored by Git).

## Repository Layout

```
src/        Python modules (currently only `anisotropy`)
examples/   Example notebooks including `benchmark.ipynb`
notebooks/  Placeholder for additional analysis notebooks
scripts/    Future command-line entry points
results/    Output CSV files (not tracked)
docs/       Static report files (PDF/HTML)
```

Raw simulation data are **not** included. Place your HDF5 snapshots and Sun-position tables under `data/` as this directory is ignored by Git.

## Running the Example Notebook

`examples/benchmark.ipynb` shows a sanitized version of the pipeline without any hardcoded local paths. You can open it in Jupyter or Google Colab once the data files are available under `data/`:

```bash
jupyter notebook examples/benchmark.ipynb
```

## Benchmark Results

The table below summarizes the average runtimes measured on Google Colab (Intel Xeon CPU and NVIDIA T4 GPU) for halos 49–70 and snapshots 102 and 121–176.

| Method                | Avg Total Runtime (s) | Std Dev (s) | Avg per Snapshot (s) | Std Dev per Snapshot (s) |
|-----------------------|----------------------:|------------:|---------------------:|-------------------------:|
| Pandas/NumPy          | 713.62                | 6.30        | 0.612                | 0.0054                   |
| Numba CPU (parallel)  | 399.49                | 2.15        | 0.343                | 0.0018                   |
| CUDA (GPU)            | 92.47                 | 1.75        | 0.079                | 0.0015                   |

## Testing

Run the unit tests with:

```bash
pytest -q
```

The current test suite only covers the `anisotropy` helper but ensures the package can be imported.

## Next Steps

1. Refactor the notebooks so that halo IDs and snapshot lists become parameters of functions under `src/`.
2. Ensure the three implementations expose a consistent API (e.g. `snapshot()` and `run_all()` helpers).
3. Provide a dispatcher script in `scripts/` that selects the baseline, CPU, or GPU pipeline at runtime.
