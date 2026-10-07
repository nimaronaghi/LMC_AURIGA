# Reproducible CPU benchmark

The benchmark measures one operation: constructing a `(12, N)` array of particle speeds relative to twelve supplied observer velocities. All backends evaluate the same Euclidean norm in float64. Available implementations are the scalar Python reference, NumPy, and optional parallel Numba. There is no GPU benchmark.

## Run

```sh
python -m lmc_auriga benchmark --output results/benchmark-numpy --counts 1000 10000 100000 --repeats 5 --seed 42 --backends numpy
```

For parallel CPU comparison:

```sh
python -m pip install -e ".[parallel]"
python -m lmc_auriga benchmark --output results/benchmark-cpu --counts 1000 10000 100000 --repeats 5 --seed 42 --backends numpy numba
```

Add `python` to `--backends` for the scalar reference, preferably starting with small counts. The command defaults to requesting NumPy and Numba; an absent optional Numba dependency is explicitly recorded as skipped. A broken installed backend raises its error instead of silently using another implementation. Use a fresh output directory for each experiment: existing benchmark CSV or JSON files are refused before warm-up or timing begins.

## Workload and timing boundary

Particle velocities are seeded independent Gaussian components with standard deviation 160 km/s. The twelve synthetic observer vectors are `(30*cos(phi), 220 + 30*sin(phi), 7)` km/s, where `phi = 2*pi*j/12`. This is a computational fixture, separate from the physical observer approximation used by `analyze`. Reinitializing the generator with the same seed gives matching particle prefixes across counts; every backend receives identical arrays at a given count.

Inputs are generated, converted to contiguous float64, and validated before timing. Each backend gets one untimed warm-up per count, including Numba compilation when required. Timed backend order rotates between repeats. `time.perf_counter` measures the kernel and output allocation; input preparation, validation, JIT warm-up, parity comparisons, HDF5 reading, spatial selection, histogramming, and file writing are outside the timer.

Every output is checked against a separately computed NumPy reference with `rtol=1e-12` and `atol=1e-12`. Maximum absolute error and maximum relative error over nonzero reference values are recorded. Independent scalar and analytic checks live in the test suite. A parity failure aborts the run before result files are written.

## Evidence and interpretation

- `raw_timings.csv` retains every repeat, particle/phase count, backend, elapsed seconds, and parity errors.
- `metadata.json` records the seed, fixture hashes, source hashes, requested/measured/skipped backends, timing scope, clock resolution, environment, CPU information, and Numba thread count when used. It includes mean, median, sample standard deviation when at least two repeats exist, minimum, and maximum timings.

Keep these files with any performance claim and identify the tested commit. Timing scatter reflects execution variability, not uncertainty in a physical model. Small workloads may be dominated by allocation, dispatch, or thread overhead; Numba is not assumed to be faster. Results do not measure end-to-end snapshot analysis or support a general hardware-independent speedup. Report absolute runtimes and workload size alongside any ratio, and compare runs with matching scope and recorded thread settings.
