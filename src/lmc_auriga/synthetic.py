"""Reproducible synthetic halo/stream fixtures, not astrophysical predictions."""

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from .io import SCHEMA_VERSION, read_metadata


def write_synthetic(path, n_particles=20000, seed=42, lmc_fraction=0.05):
    """Create a processed HDF5 toy snapshot and return its validated metadata.

    Positions are uniform in volume in the 4--14 kpc shell, with isotropic
    directions. Host velocities are isotropic N(0,155^2) km/s; tagged stream
    velocities have mean (0,-300,80) and dispersions (70,50,50) km/s. All
    weights are one. The tagged count is nearest-integer N*f, with ties up.

    These prescribed kinematics are neither a gravitational simulation nor
    an inferred LMC model. The Gaussians have no imposed escape-speed cutoff.
    The local NumPy RNG does not change global random state. The destination
    must not already exist, protecting user-supplied observational/simulation
    files from accidental replacement. Parent directories are created.
    """
    if isinstance(n_particles, bool) or not isinstance(n_particles, Integral) or n_particles < 0:
        raise ValueError("n_particles must be a nonnegative integer")
    if (isinstance(seed, bool) or not isinstance(seed, Integral)
            or not 0 <= seed <= np.iinfo(np.uint64).max):
        raise ValueError("seed must be an integer from 0 through 2**64-1")
    if (isinstance(lmc_fraction, bool) or not isinstance(lmc_fraction, (int, float))
            or not math.isfinite(lmc_fraction) or not 0 <= lmc_fraction <= 1):
        raise ValueError("lmc_fraction must be finite and between 0 and 1")
    n_particles, seed = int(n_particles), int(seed)
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to replace existing snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    parameters = {
        "n_particles": n_particles, "seed": seed,
        "lmc_fraction_requested": float(lmc_fraction),
        "radial_shell_kpc": [4.0, 14.0],
        "position_distribution": "uniform in shell volume; isotropic directions",
        "mw_mean_km_s": [0.0, 0.0, 0.0],
        "mw_dispersion_km_s": [155.0, 155.0, 155.0],
        "lmc_mean_km_s": [0.0, -300.0, 80.0],
        "lmc_dispersion_km_s": [70.0, 50.0, 50.0],
        "weight_per_particle": 1.0,
        "tagged_count_rounding": "nearest integer; half ties round upward",
    }
    rng = np.random.default_rng(seed)
    n_lmc = min(n_particles, math.floor(n_particles * lmc_fraction + 0.5))
    tags = np.full(n_particles, -1, dtype=np.int8)
    tags[:n_lmc] = 1
    rng.shuffle(tags)
    inner, outer = parameters["radial_shell_kpc"]
    radius = np.cbrt(inner**3 + rng.random(n_particles) * (outer**3 - inner**3))
    cos_theta = rng.uniform(-1, 1, n_particles)
    phi = rng.uniform(0, 2 * np.pi, n_particles)
    sin_theta = np.sqrt(np.maximum(0, 1 - cos_theta**2))
    coordinates = radius[:, None] * np.column_stack(
        (sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))
    velocities = rng.normal(parameters["mw_mean_km_s"], parameters["mw_dispersion_km_s"],
                            size=(n_particles, 3))
    velocities[tags == 1] = rng.normal(parameters["lmc_mean_km_s"],
                                      parameters["lmc_dispersion_km_s"], size=(n_lmc, 3))
    weights = np.ones(n_particles, dtype=np.float64)
    # Exclusive creation also protects against a destination appearing between
    # the existence check and file creation. No input file is ever opened to write.
    with h5py.File(destination, "x") as snapshot:
        for name, values in (("Coordinates", coordinates), ("Velocities", velocities),
                             ("LMC_tag", tags), ("Weights", weights)):
            snapshot.create_dataset(name, data=values, track_times=False)
        snapshot.attrs.update({
            "schema_version": SCHEMA_VERSION,
            "position_unit": "kpc", "velocity_unit": "km/s", "frame": "galactocentric",
            "provenance": "synthetic",
            "source": "Prescribed Gaussian halo plus tagged stream; generated locally, no Auriga data used",
            "model_scope": "Toy kinematic fixture; not a gravitational simulation, equilibrium model, "
                           "escape-truncated distribution, or prediction of the real LMC",
            "generator": "lmc_auriga.synthetic.write_synthetic",
            "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__, "h5py_version": h5py.__version__,
            "random_generator": type(rng.bit_generator).__name__,
            "seed": np.uint64(seed), "n_particles": n_particles,
            "lmc_count": n_lmc, "mw_count": n_particles - n_lmc,
            "lmc_fraction_requested": float(lmc_fraction),
            "lmc_fraction_realized": n_lmc / n_particles if n_particles else 0.0,
            "weighting": "equal particle weights; number-weighted synthetic demonstration",
            "parameters_json": json.dumps(parameters, sort_keys=True, allow_nan=False),
        })
    return read_metadata(destination)
