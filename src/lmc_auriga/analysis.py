"""Chunked, weighted halo observables with explicit normalization and provenance."""

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from . import __version__
from .backends import shifted_speeds
from .io import iter_snapshot, read_metadata
from .physics import observer_velocity, selection_mask, spherical_velocities

COMPONENTS = ("total", "mw", "lmc")


class _Moments:
    """Merge weighted moments relative to one observed velocity reference."""

    def __init__(self):
        self.n = 0
        self.weight = 0.0
        self._anchor = None
        self._mean_offset = np.zeros(3)
        self.m2 = np.zeros(3)

    @property
    def mean(self):
        if self._anchor is None:
            return np.zeros(3)
        return self._anchor + self._mean_offset

    def update(self, values, weights):
        if not len(values):
            return
        if self._anchor is None:
            self._anchor = values[0].copy()
        # Keep the same observed reference across chunks. A constant component
        # then has exactly zero offsets, even if its weighted absolute mean
        # would round away from the observed value. Small real dispersions
        # remain measurable without an arbitrary zero-variance tolerance.
        offsets = values - self._anchor
        weight = float(weights.sum())
        mean = np.average(offsets, axis=0, weights=weights)
        m2 = np.sum(weights[:, None] * (offsets - mean) ** 2, axis=0)
        delta = mean - self._mean_offset
        combined = self.weight + weight
        self.m2 += m2 + delta ** 2 * (self.weight / combined) * weight
        self._mean_offset += delta * (weight / combined)
        self.weight = combined
        self.n += len(values)

    def beta(self):
        if self.n < 2 or self.m2[0] <= 0:
            return None
        return float(1 - (self.m2[1] + self.m2[2]) / (2 * self.m2[0]))


def _tail_sums(speeds, weights, thresholds):
    """Unnormalized strict-tail and inverse-speed sums in O(N log N + K log N)."""
    order = np.argsort(speeds)
    speeds, weights = speeds[order], weights[order]
    inverse = np.divide(weights, speeds, out=np.zeros_like(weights), where=speeds > 0)
    tail = np.r_[np.cumsum(weights[::-1])[::-1], 0.0]
    eta = np.r_[np.cumsum(inverse[::-1])[::-1], 0.0]
    index = np.searchsorted(speeds, thresholds, side="right")
    return tail[index], eta[index]


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def analyze_snapshot(path, output, *, shell=(6.0, 10.0), cone_axis=None,
                     half_angle_deg=45.0, vc=220.0, basis=None, phase_count=12,
                     chunk_size=100000, speed_max=1000.0, bin_width=20.0,
                     backend="numpy"):
    """Analyze one schema-v1 snapshot without loading all particles into memory.

    Each component is normalized by the TOTAL selected weight, so MW + LMC
    contributions equal the total. Phase means are applied to observables,
    never to each particle's speed before histogramming. Beta is calculated
    once in the host frame from centered, weighted spherical components.
    Output files are refused if any already exists; input is never modified.
    """
    if isinstance(phase_count, bool) or not isinstance(phase_count, int) or phase_count < 1:
        raise ValueError("phase_count must be a positive integer")
    if not np.isfinite([speed_max, bin_width]).all() or min(speed_max, bin_width) <= 0:
        raise ValueError("speed_max and bin_width must be finite and positive")
    if not np.isclose(speed_max / bin_width, round(speed_max / bin_width)):
        raise ValueError("speed_max must be an integer multiple of bin_width")
    if speed_max < bin_width:
        raise ValueError("speed_max must be at least bin_width")
    # Validate even an empty snapshot's selection configuration.
    selection_mask(np.array([[1., 0., 0.]]), shell, cone_axis, half_angle_deg)
    phases = np.arange(phase_count) / phase_count
    observers = np.array([observer_velocity(p, vc=vc, basis=basis) for p in phases])
    shifted_speeds(np.zeros((1, 3)), observers, backend=backend)
    metadata = read_metadata(path)
    output = Path(output)
    filenames = ("speed_distribution.csv", "observables.csv", "anisotropy.csv", "report.json")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("analysis output already exists; choose a fresh directory")
    edges = np.linspace(0, speed_max, round(speed_max / bin_width) + 1)
    thresholds = edges.copy()
    hist = np.zeros((3, len(edges) - 1))
    tails = np.zeros((3, len(thresholds)))
    etas = np.zeros_like(tails)
    overflow = np.zeros(3)
    moments = [_Moments() for _ in COMPONENTS]
    for positions, velocities, tags, weights in iter_snapshot(path, chunk_size=chunk_size):
        selected = selection_mask(positions, shell, cone_axis, half_angle_deg)
        if not np.any(selected):
            continue
        positions, velocities, tags, weights = (a[selected] for a in (positions, velocities, tags, weights))
        spherical = spherical_velocities(positions, velocities)
        speeds = shifted_speeds(velocities, observers, backend=backend)
        for component, mask in enumerate((np.ones(len(tags), dtype=bool), tags == -1, tags == 1)):
            w = weights[mask]
            moments[component].update(spherical[mask], w)
            for phase_speeds in speeds[:, mask]:
                hist[component] += np.histogram(phase_speeds, bins=edges, weights=w)[0]
                tail, eta = _tail_sums(phase_speeds, w, thresholds)
                tails[component] += tail
                etas[component] += eta
                overflow[component] += w[phase_speeds > speed_max].sum()
    if not moments[0].n:
        raise ValueError("selection contains no particles")
    normalization = moments[0].weight * phase_count
    hist /= normalization * np.diff(edges)
    tails /= normalization
    etas /= normalization
    overflow /= normalization
    if not all(np.isfinite(a).all() for a in (hist, tails, etas, overflow)):
        raise ValueError("observable accumulation exceeds float64 range")
    rows = [{"component": label, "n": m.n, "weight": m.weight,
             "beta": m.beta(), "mean_spherical_km_s": m.mean.tolist() if m.n else None}
            for label, m in zip(COMPONENTS, moments)]
    versions = {}
    for name in ("numpy", "h5py", "numba", "matplotlib", "pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass  # Optional packages need not be installed for analysis.
    report = {
        "schema_version": 1, "package_version": __version__,
        "input_sha256": _hash_file(path), "input_metadata": metadata,
        "selected_count": moments[0].n, "selected_weight": moments[0].weight,
        "configuration": {"shell": list(shell), "cone_axis": None if cone_axis is None else list(cone_axis),
            "half_angle_deg": half_angle_deg, "vc_km_s": vc, "basis_rows": np.eye(3).tolist() if basis is None else np.asarray(basis).tolist(),
            "phase_count": phase_count, "phases": phases.tolist(), "chunk_size": chunk_size,
            "speed_max_km_s": speed_max, "bin_width_km_s": bin_width, "backend": backend},
        "normalization": "All components divided by total selected weight; arithmetic mean over equally spaced annual phases.",
        "weighting": "supplied positive weights" if metadata["has_weights"] else "particle number (unit weights)",
        "histogram_overflow_fraction": dict(zip(COMPONENTS, overflow.tolist())),
        "anisotropy": rows,
        "scope": "Kinematic snapshot analysis; no detector response, local density inference, or independent-sample uncertainty estimate.",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "dependencies": versions},
        "source_sha256": {p.name: _hash_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    serialized = json.dumps(report, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / filenames[0], ["speed_left_km_s", "speed_right_km_s", "total_pdf", "mw_pdf", "lmc_pdf"],
               zip(edges[:-1], edges[1:], *hist))
    _write_csv(output / filenames[1], ["vmin_km_s", "total_tail", "mw_tail", "lmc_tail", "total_eta", "mw_eta", "lmc_eta"],
               zip(thresholds, *tails, *etas))
    _write_csv(output / filenames[2], ["component", "n", "weight", "beta"],
               ((r["component"], r["n"], r["weight"], r["beta"]) for r in rows))
    (output / filenames[3]).write_text(serialized, encoding="utf-8")
    return report
