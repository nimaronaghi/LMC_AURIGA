# Local CPU measurement

These raw timings were generated on one Windows PC using the package's synthetic
float64 fixture: 12 observer vectors, 1,000/10,000/100,000 particles, and five
repeats per backend. `metadata.json` records the environment, thread count,
source hashes, warm-up policy, numerical parity, and timing summaries.

Reproduce the protocol with a fresh destination:

```sh
python -m lmc_auriga benchmark --output results/benchmark-repeat --counts 1000 10000 100000 --repeats 5 --seed 42 --backends numpy numba
```

The workload measures the observer-speed kernel including allocation. It excludes
I/O, spatial selection, statistical reduction, JIT compilation, and plotting.
These measurements are machine-specific evidence, not a general speedup claim
or a reproduction of the historical notebook's timings.
