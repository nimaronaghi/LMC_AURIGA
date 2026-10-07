"""Validated, streaming access to processed halo-particle HDF5 snapshots.

This schema contains already converted physical coordinates and velocities.
It does not interpret raw Auriga/Gadget snapshots, cosmological scale factors,
or undocumented observer frames. Unit and frame declarations are mandatory.
"""

from numbers import Integral

import h5py
import numpy as np


SCHEMA_VERSION = 1
REQUIRED_ATTRIBUTES = {
    "position_unit": "kpc",
    "velocity_unit": "km/s",
    "frame": "galactocentric",
}


def _metadata_value(value):
    """Convert HDF5 scalar/array attributes into finite JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return _metadata_value(value.tolist())
    if isinstance(value, np.generic):
        return _metadata_value(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("metadata attributes must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported metadata attribute type: {type(value).__name__}")


def _validated_header(snapshot):
    attributes = {name: _metadata_value(value) for name, value in snapshot.attrs.items()}
    version = attributes.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be the integer {SCHEMA_VERSION}")
    for name, expected in REQUIRED_ATTRIBUTES.items():
        if attributes.get(name) != expected:
            raise ValueError(f"{name} must be explicitly declared as {expected!r}; "
                             "raw snapshot conversion is not automatic")
    provenance = attributes.get("provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must be a nonempty string describing the data source")

    required = ("Coordinates", "Velocities", "LMC_tag")
    for name in required:
        if name not in snapshot or not isinstance(snapshot[name], h5py.Dataset):
            raise ValueError(f"required root dataset {name!r} is missing")
    coordinates, velocities, tags = (snapshot[name] for name in required)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("Coordinates must have shape (N, 3)")
    count = coordinates.shape[0]
    if velocities.shape != (count, 3):
        raise ValueError("Velocities must have shape (N, 3) matching Coordinates")
    if tags.shape != (count,):
        raise ValueError("LMC_tag must have shape (N,) matching Coordinates")
    names = list(required)
    if "Weights" in snapshot:
        if not isinstance(snapshot["Weights"], h5py.Dataset) or snapshot["Weights"].shape != (count,):
            raise ValueError("Weights must have shape (N,) matching Coordinates")
        names.append("Weights")
    for name in names:
        if snapshot[name].dtype.kind not in "fiu":
            raise ValueError(f"{name} must contain real numeric values")
    attributes["n_particles"] = count
    attributes["has_weights"] = "Weights" in snapshot
    return attributes


def read_metadata(path):
    """Read JSON-safe metadata and validate the header without loading particles.

    Shapes, numeric dtypes, units, frame, schema version and provenance are
    checked here. Particle values are checked when each chunk is consumed by
    :func:`iter_snapshot`, keeping validation memory bounded.
    """
    with h5py.File(path, "r") as snapshot:
        return _validated_header(snapshot)


def iter_snapshot(path, chunk_size=100000):
    """Yield validated ``coordinates, velocities, tags, weights`` array chunks.

    Coordinates/velocities/weights are float64; canonical tags are int8 with
    -1 for the host (MW) and +1 for the tagged (LMC) population. Unknown tags
    are rejected, never silently treated as LMC. Missing weights give ones.
    Weights must be finite and strictly positive. An empty valid snapshot
    yields no chunks. No more than one chunk per field is loaded at once.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    with h5py.File(path, "r") as snapshot:
        metadata = _validated_header(snapshot)
        for start in range(0, metadata["n_particles"], int(chunk_size)):
            stop = min(start + int(chunk_size), metadata["n_particles"])
            coordinates = np.asarray(snapshot["Coordinates"][start:stop], dtype=np.float64)
            velocities = np.asarray(snapshot["Velocities"][start:stop], dtype=np.float64)
            tags = np.asarray(snapshot["LMC_tag"][start:stop])
            weights = (np.asarray(snapshot["Weights"][start:stop], dtype=np.float64)
                       if metadata["has_weights"] else np.ones(stop - start, dtype=np.float64))
            for name, values in (("Coordinates", coordinates), ("Velocities", velocities),
                                 ("LMC_tag", tags), ("Weights", weights)):
                if not np.isfinite(values).all():
                    raise ValueError(f"{name} contains nonfinite values in rows [{start}, {stop})")
            if not np.isin(tags, (-1, 1)).all():
                raise ValueError(f"LMC_tag must contain only -1 (MW) or +1 (LMC); "
                                 f"invalid tag in rows [{start}, {stop})")
            if (weights <= 0).any():
                raise ValueError(f"Weights must be positive in rows [{start}, {stop})")
            yield coordinates, velocities, tags.astype(np.int8), weights
