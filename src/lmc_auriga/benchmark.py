"""Reproducible compute-only timings of equivalent observer-speed kernels."""

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from . import backends as speed_backends


SAMPLE_FIELDS = (
    "n_particles", "n_phases", "backend", "repeat", "seconds",
    "max_absolute_error", "max_relative_error",
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _fixture(count, seed):
    # Reinitializing the generator makes particle prefixes identical across N.
    velocities = np.random.default_rng(seed).normal(0.0, 160.0, size=(count, 3))
    phases = 2 * np.pi * np.arange(12) / 12
    observers = np.column_stack((30 * np.cos(phases), 220 + 30 * np.sin(phases),
                                 np.full(12, 7.0)))
    return speed_backends._prepare_inputs(velocities, observers)


def _fixture_hash(velocities, observers):
    digest = hashlib.sha256()
    digest.update(json.dumps([list(velocities.shape), list(observers.shape)]).encode("ascii"))
    digest.update(velocities.astype("<f8", copy=False).tobytes(order="C"))
    digest.update(observers.astype("<f8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _parity(result, reference):
    if result.shape != reference.shape or not np.isfinite(result).all():
        raise ArithmeticError("backend returned invalid speed data")
    difference = np.abs(result - reference)
    absolute = float(np.max(difference, initial=0.0))
    nonzero = reference != 0
    relative = float(np.max(difference[nonzero] / reference[nonzero], initial=0.0))
    if not np.allclose(result, reference, rtol=1e-12, atol=1e-12):
        raise ArithmeticError(f"backend failed numerical parity (maximum absolute error {absolute})")
    return absolute, relative


def run_benchmark(output, counts=(1000, 10000, 100000), repeats=5, seed=42,
                  backends=("numpy", "numba")):
    """Save raw timing samples and metadata; return a result dictionary.

    Timed calls operate on identical prevalidated float64 inputs and include
    output allocation. Input generation/validation, JIT warm-up, parity checks
    and I/O are excluded. Every available requested backend gets one untimed
    warm-up at each size; timed backend order rotates between repeats.

    Missing optional Numba is explicitly recorded in ``skipped_backends``.
    No GPU implementation or end-to-end HDF5 speedup is claimed. The returned
    mapping contains ``metadata``, ``samples``, ``summary`` and file ``paths``.
    Existing output files are refused before any kernel is initialized or timed.
    """
    counts = [_positive_integer(count, "particle count") for count in counts]
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("counts must be nonempty and unique")
    repeats = _positive_integer(repeats, "repeats")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if isinstance(backends, str):
        raise ValueError("backends must be a sequence of backend names")
    requested = list(backends)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("backends must be nonempty and unique")
    if any(name not in speed_backends.BACKENDS for name in requested):
        raise ValueError(f"backends must be chosen from {speed_backends.BACKENDS}")
    destination = Path(output)
    csv_path, json_path = destination / "raw_timings.csv", destination / "metadata.json"
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("benchmark output already exists; choose a fresh directory")
    available = speed_backends.available_backends()
    active = [name for name in requested if name in available]
    skipped = [{"backend": name, "reason": "optional dependency is not installed"}
               for name in requested if name not in available]
    kernels = {name: speed_backends._get_kernel(name) for name in active}
    samples, summary, fixtures = [], [], []
    for count in counts if active else ():
        velocities, observers = _fixture(count, seed)
        fixtures.append({"n_particles": count, "sha256": _fixture_hash(velocities, observers)})
        reference = speed_backends._numpy_kernel(velocities, observers)
        for name in active:
            _parity(kernels[name](velocities, observers), reference)
        for repeat in range(1, repeats + 1):
            offset = (repeat - 1) % len(active)
            for name in active[offset:] + active[:offset]:
                start = time.perf_counter()
                result = kernels[name](velocities, observers)
                elapsed = time.perf_counter() - start
                absolute, relative = _parity(result, reference)
                samples.append({
                    "n_particles": count, "n_phases": len(observers),
                    "backend": name, "repeat": repeat, "seconds": elapsed,
                    "max_absolute_error": absolute, "max_relative_error": relative,
                })
        for name in active:
            group = [row for row in samples if row["n_particles"] == count and row["backend"] == name]
            times = [row["seconds"] for row in group]
            summary.append({
                "n_particles": count, "n_phases": len(observers), "backend": name,
                "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
                "stdev_seconds": statistics.stdev(times) if repeats > 1 else None,
                "min_seconds": min(times), "max_seconds": max(times),
                "max_absolute_error": max(row["max_absolute_error"] for row in group),
                "max_relative_error": max(row["max_relative_error"] for row in group),
            })
    environment = {
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if "numba" in active:
        import numba
        environment.update(numba=importlib.metadata.version("numba"), numba_threads=numba.get_num_threads())
    metadata = {
        "schema_version": 1, "status": "completed" if active else "no_available_backends",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "particle_counts": counts, "repeats": repeats, "seed": seed, "n_phases": 12,
        "requested_backends": requested, "measured_backends": active, "skipped_backends": skipped,
        "dtype": "float64", "velocity_unit": "km/s (synthetic fixture)",
        "fixture": "Gaussian particle velocities with component sigma=160; synthetic observer "
                   "velocities (30*cos(phi), 220+30*sin(phi), 7), phi=2*pi*j/12",
        "fixtures": fixtures,
        "fixture_hash_encoding": "SHA256 of JSON velocity/observer shapes then C-order little-endian float64 arrays",
        "timing_scope": "speed kernel on prevalidated float64 arrays; includes output allocation; "
                        "excludes input generation/validation, JIT warm-up, parity comparison and I/O",
        "warmup_calls_per_backend_and_size": 1,
        "backend_order": "rotate starting backend each repeat",
        "reference": "NumPy float64 kernel; separate untimed reference evaluation",
        "parity_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "clock": "time.perf_counter", "clock_resolution_seconds": time.get_clock_info("perf_counter").resolution,
        "backend_source_sha256": hashlib.sha256(Path(speed_backends.__file__).read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment,
    }
    destination.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(samples)
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(metadata, summary=summary, samples_file=csv_path.name),
                                indent=2, allow_nan=False) + "\n")
    return {"metadata": metadata, "samples": samples, "summary": summary,
            "paths": {"samples": str(csv_path), "metadata": str(json_path)}}
