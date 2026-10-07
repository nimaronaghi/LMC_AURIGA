"""Explicit, lossless unit/frame declaration for confirmed processed legacy data.

This importer does not convert raw cosmological snapshots. Coordinates and
velocities are copied unchanged only after the caller explicitly confirms
host-centered physical kpc and host-rest km/s.
"""

import base64
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np

from . import __version__
from .io import SCHEMA_VERSION, read_metadata


def _original_attribute(value):
    """Preserve ordinary attributes, with tagged encodings for non-JSON values."""
    if isinstance(value, np.ndarray):
        return _original_attribute(value.tolist())
    if isinstance(value, np.generic):
        return _original_attribute(value.item())
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [_original_attribute(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"type": "nonfinite_float", "value": str(value)}
    if isinstance(value, complex):
        return {"type": "complex", "real": _original_attribute(value.real),
                "imaginary": _original_attribute(value.imag)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"cannot preserve original attribute of type {type(value).__name__} as JSON")


def _source_header(snapshot):
    names = ["Coordinates", "Velocities", "LMC_tag"]
    if "Weights" in snapshot:
        names.append("Weights")
    for name in names:
        if name not in snapshot or not isinstance(snapshot[name], h5py.Dataset):
            raise ValueError(f"legacy root dataset {name!r} is required")
        if snapshot[name].dtype.kind not in "fiu":
            raise ValueError(f"legacy dataset {name!r} must contain real numeric values")
    coordinates = snapshot["Coordinates"]
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("legacy Coordinates must have shape (N, 3)")
    count = coordinates.shape[0]
    if snapshot["Velocities"].shape != (count, 3):
        raise ValueError("legacy Velocities must have shape (N, 3) matching Coordinates")
    for name in names[2:]:
        if snapshot[name].shape != (count,):
            raise ValueError(f"legacy {name} must have shape (N,) matching Coordinates")
    return names, count


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_no_replace(temporary, destination):
    """Publish a completed file atomically, failing if the destination exists."""
    if os.name == "nt":
        # Windows rename never replaces an existing destination.
        os.rename(temporary, destination)
    else:
        # POSIX rename replaces files; link instead atomically creates the new
        # name or fails with EEXIST. Both paths are on the same filesystem.
        os.link(temporary, destination)
        temporary.unlink()


def import_legacy_snapshot(source, destination, *, provenance,
                           confirmed_processed_frame=False, chunk_size=100000):
    """Import a confirmed processed legacy snapshot without unit conversions.

    The confirmation must be the boolean True. Legacy tags -1 identify the
    host; strictly positive tags identify LMC material. Zero, other negatives
    and nonfinite tags are rejected. ``Original_LMC_tag`` retains exact original
    tag values and dtype. Existing positive ``Weights`` are preserved; absent
    weights remain absent, representing unit number weights in the reader.

    Coordinates, velocities and weights retain their original numeric values,
    dtypes and dataset attributes. Original root attributes are preserved as
    ``original_attributes_json``. Every particle is validated in bounded chunks
    before an exclusively created final filename becomes visible. Neither the
    source nor an existing destination is overwritten. Return validated metadata.
    """
    if confirmed_processed_frame is not True:
        raise ValueError("explicit confirmed_processed_frame=True is required: coordinates must already "
                         "be host-centered physical kpc and velocities host-rest km/s")
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("provenance must be a nonempty description of the source data")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    # No final-path handle lookup is needed: all existing destinations,
    # including symlinks and hard-link aliases, are refused below and at publish.
    source = Path(os.path.abspath(source))
    destination = Path(os.path.abspath(destination))
    if not source.is_file():
        raise ValueError("source must be an existing HDF5 file")
    if os.path.normcase(source) == os.path.normcase(destination):
        raise ValueError("source and destination must be different files")
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace existing destination: {destination}")

    temporary = None
    try:
        with h5py.File(source, "r") as legacy:
            names, count = _source_header(legacy)
            original = json.dumps({name: _original_attribute(value) for name, value in legacy.attrs.items()},
                                  sort_keys=True, allow_nan=False)
            source_hash = _file_sha256(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                                     dir=destination.parent)
            os.close(descriptor)
            temporary = Path(filename)
            with h5py.File(temporary, "w") as target:
                for name in names:
                    original_dataset = legacy[name]
                    if name == "LMC_tag":
                        copied = target.create_dataset("Original_LMC_tag", shape=original_dataset.shape,
                                                       dtype=original_dataset.dtype, chunks=True, track_times=False)
                        target.create_dataset("LMC_tag", shape=(count,), dtype=np.int8,
                                              chunks=True, track_times=False)
                    else:
                        copied = target.create_dataset(name, shape=original_dataset.shape,
                                                       dtype=original_dataset.dtype, chunks=True, track_times=False)
                    for attribute, value in original_dataset.attrs.items():
                        copied.attrs[attribute] = value
                host_count = lmc_count = 0
                for start in range(0, count, int(chunk_size)):
                    stop = min(start + int(chunk_size), count)
                    values = {name: legacy[name][start:stop] for name in names}
                    for name, data in values.items():
                        if not np.isfinite(data).all():
                            raise ValueError(f"legacy {name} contains nonfinite values in rows [{start}, {stop})")
                    tags = values["LMC_tag"]
                    if not ((tags == -1) | (tags > 0)).all():
                        raise ValueError(f"legacy LMC_tag must be -1 (host) or strictly positive (LMC); "
                                         f"invalid value in rows [{start}, {stop})")
                    if "Weights" in values and (values["Weights"] <= 0).any():
                        raise ValueError(f"legacy Weights must be positive in rows [{start}, {stop})")
                    for name, data in values.items():
                        target["Original_LMC_tag" if name == "LMC_tag" else name][start:stop] = data
                    canonical_tags = np.where(tags == -1, -1, 1).astype(np.int8)
                    target["LMC_tag"][start:stop] = canonical_tags
                    host_count += int(np.count_nonzero(canonical_tags == -1))
                    lmc_count += int(np.count_nonzero(canonical_tags == 1))
                target.attrs.update({
                    "schema_version": SCHEMA_VERSION, "position_unit": "kpc", "velocity_unit": "km/s",
                    "frame": "galactocentric", "provenance": provenance.strip(),
                    "confirmed_processed_frame": True,
                    "frame_assertion": "User confirmed host-centered physical kpc coordinates and host-rest km/s velocities",
                    "source_file_name": source.name, "source_file_sha256": source_hash,
                    "import_method": "lmc_auriga.importing.import_legacy_snapshot; chunked copy without coordinate or velocity conversion",
                    "import_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "package_version": __version__,
                    "tag_mapping": "legacy -1 -> canonical -1 (MW); legacy >0 -> canonical +1 (LMC); originals preserved",
                    "weight_policy": ("existing positive Weights copied unchanged" if "Weights" in names
                                      else "unit number weights; legacy source has no Weights dataset"),
                    "original_attributes_json": original,
                    "mw_count": host_count, "lmc_count": lmc_count,
                })
                target.flush()
        # Header validation occurs before publication; every data chunk was
        # already checked above. No consumer sees a partially validated file.
        metadata = read_metadata(temporary)
        _publish_no_replace(temporary, destination)
        return metadata
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
