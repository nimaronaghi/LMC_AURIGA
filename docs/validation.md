# Validation and historical corrections

Validation combines analytic examples, geometric invariants, hand-computed weighted statistics, malformed input fixtures, and agreement between independently structured kernels. These checks establish implementation behavior; they do not validate an astronomical interpretation of undocumented data.

## Run the checks

```sh
python -m pip install -e ".[plot,test]"
python -m pytest -q
```

To exercise the optional parallel kernel, install `python -m pip install -e ".[parallel]"` and rerun the suite. Optional-dependency tests are skipped when their dependency is unavailable. Test results should be reported with the tested revision and environment; the documentation does not freeze a test count.

| Area | Independent checks | Tests |
| --- | --- | --- |
| Geometry and frames | Known vectors and poles; speed preservation; radial projection under rotation; inclusive selection boundaries; observer phase, basis direction, and periodicity | [test_physics.py](../tests/test_physics.py) |
| Statistical definitions | Known anisotropy and streaming offsets; exact weighted inverse-speed atoms; Maxwell-distribution quadrature; strict zero/threshold convention | [test_physics.py](../tests/test_physics.py) |
| Analysis pipeline | Hand-calculated two-phase distributions; component additivity; weighted moments across chunk sizes; histogram overflow; host-frame beta independent of observer speed | [test_analysis.py](../tests/test_analysis.py) |
| Processed input | Shape, dtype, units, version and tag rejection; empty input; invalid values in later chunks; unit-weight fallback | [test_io.py](../tests/test_io.py) |
| Synthetic input | Identical seeded arrays; distinct seeds; counts, radial-volume distribution, and velocity moments; refusal to replace an existing file | [test_synthetic.py](../tests/test_synthetic.py) |
| CPU kernels | Scalar-reference parity, analytic relative speeds, retained phases, noncontiguous inputs, input immutability, and ordered parallel output | [test_backends.py](../tests/test_backends.py) |
| Benchmark protocol | Identical inputs; warm-up excluded from samples; CSV/metadata round trip; explicit missing backend; failed parity rejected | [test_benchmark.py](../tests/test_benchmark.py) |
| Figures | Input tables unchanged; physical canvas size; component additivity; zero handling; plotting style restored after export failure | [test_plotting.py](../tests/test_plotting.py) |

The legacy `src/anisotropy.py` adapter also has explicit schema tests in [test_anisotropy.py](../tests/test_anisotropy.py). New callers should use the named-array APIs in `lmc_auriga`.

## Corrections to the historical workflow

The original [benchmark notebook](legacy/benchmark-original.ipynb.txt) and [analysis fragments](legacy/lmc-analysis-fragments.txt) are preserved as text for provenance. They are archival material, not runnable entry points or a supported benchmark specification.

| Historical issue | Supported implementation |
| --- | --- |
| Spherical polar projection did not preserve speed for a simple off-axis vector | Explicit orthonormal spherical basis, with pole convention and speed-preservation tests |
| Implicit positional-column layouts could select the wrong velocity columns | Named arrays with explicit shapes; legacy adapter validates its declared layout |
| Different benchmark paths used different selection/observer operations | One shared selection and observer calculation; equivalent speed kernels receive identical arrays |
| Particle speeds were averaged over phases before constructing nonlinear observables | Histograms, strict tails, and inverse-speed sums are evaluated per phase and then averaged |
| Parallel compaction depended on a shared output counter | Each parallel iteration owns distinct particle output entries; no shared compaction counter |
| Component normalization could obscure the component's contribution to the total | One selected total-weight denominator for all speed observables |
| Hard-coded paths and implicit physical conventions prevented portable reproduction | CLI configuration, schema validation, seeded fixture, source/input hashes, and recorded parameters |
| Runtime summaries lacked an auditable common workload | Compute-only repeated samples with warm-up, numerical parity, explicit timing scope, and environment metadata |

These corrections concern repository code and its supported workflow. They establish no change to, reproduction of, or error in any historical paper's scientific results. The project contains no validated GPU implementation or substantiated CUDA performance comparison.

## Limits and checks before scientific use

Numerical agreement is necessary but not sufficient for physical validity. Confirm units, origin, velocity rest frame, observer orientation, component tagging, particle weighting, and the intended spatial sample. Assess sensitivity to selection, phase count, and resolution. Correlated simulation particles and repeated observer phases must not be treated as independent measurements when constructing uncertainty estimates.

The public showcase is a software and methods demonstration. It cannot establish equilibrium, the local dark-matter density, an LMC-induced change in a real galaxy, or a detector event rate. The [methods](methods.md) and [data contract](data_contract.md) identify the assumptions required for a future documented simulation application.
