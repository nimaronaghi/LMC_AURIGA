# Processed particle-data contract

`lmc_auriga.io` reads a deliberately small, versioned HDF5 format. It describes particles whose conversion, centering, rest frame, and component assignment have already been established. Passing structural validation does not verify those scientific declarations.

## Schema version 1

Required root attributes are exact and case-sensitive:

| Attribute | Required value |
| --- | --- |
| `schema_version` | Integer `1` |
| `position_unit` | String `kpc`, meaning physical kiloparsecs |
| `velocity_unit` | String `km/s` |
| `frame` | String `galactocentric` |
| `provenance` | Nonempty source description; the supplied toy generator uses `synthetic` |

Root datasets share the same row count `N`:

| Dataset | Shape | Interpretation |
| --- | --- | --- |
| `Coordinates` | `(N, 3)` | Cartesian `(x, y, z)` relative to the declared galaxy center |
| `Velocities` | `(N, 3)` | Cartesian `(vx, vy, vz)` in the corresponding host rest frame |
| `LMC_tag` | `(N,)` | Exactly `-1` for the host/MW component and `+1` for the tagged/LMC component |
| `Weights` | `(N,)`, optional | Finite, strictly positive particle weights; missing weights become ones |

All datasets must contain real numeric values. Every consumed chunk is checked for finite coordinates and velocities, valid tags, and valid weights. Other tag encodings, including continuous membership fields, require an explicit documented conversion. A filename or plausible numeric range never establishes units or component membership.

Weights define the empirical measure. Unit weights give number-weighted results; documented particle masses give mass-weighted results. The program does not infer equal masses, a mass unit, or an absolute local density. Record the weight meaning and any conversions in additional provenance attributes. All components share the total selected weight for speed observables; each component uses its own centered moments for anisotropy.

## Reader behavior

```python
from lmc_auriga.io import iter_snapshot, read_metadata

metadata = read_metadata("data/processed/snapshot.hdf5")
for positions, velocities, tags, weights in iter_snapshot(
    "data/processed/snapshot.hdf5", chunk_size=100_000
):
    # Each tuple contains at most chunk_size rows.
    pass
```

`read_metadata` validates attributes, shapes, numeric dtypes, and matching lengths without scanning particle values. It returns JSON-compatible attributes plus `n_particles` and `has_weights`. `iter_snapshot` repeats the structural check and validates values as it reads; an invalid late chunk raises an error rather than being dropped. Empty snapshots are valid containers, but analysis requires a nonempty spatial selection. Unknown additional datasets are not used.

## Frame and provenance record

For a scientific input, retain the original source and checksum, simulation/snapshot identity, scale factor and cosmology when relevant, particle type, selection history, position and velocity conversion factors, center and bulk-velocity subtraction, axis transformations, tag definition, and weight meaning. Keep this record with the processed data. Required schema strings are declarations, not substitutes for that evidence.

The observer basis supplied to analysis is separate: its rows are the local Galactic `U,V,W` axes expressed in the processed coordinate system. The default identity basis is appropriate only when those axes already coincide. Positions and velocities must use the same Cartesian axis convention. See [the observer model](methods.md#observer-frame-speeds).

## From raw Auriga data

Raw Auriga output is a different format. The [official specification](https://wwwmpa.mpa-garching.mpg.de/auriga/dataspecs.html) documents particle-type groups, coordinates in comoving Mpc/h, and the scale-factor conversion for stored velocities. It also notes the public analysis package's `[Z,Y,X]` vector ordering. Check the exact source and loading route before exporting `(x,y,z)` columns.

For raw positions, physical displacement conversion involves the scale factor, Hubble parameter, Mpc-to-kpc factor, and a documented galaxy center; periodic wrapping may also matter. For velocities, apply the documented stored-to-peculiar conversion and a consistent host bulk-velocity subtraction. These steps are not implemented automatically here. The [official analysis examples](https://wwwmpa.mpa-garching.mpg.de/auriga/analysis.html) provide the starting point for loading, centering, and selecting particles. The [release page](https://wwwmpa.mpa-garching.mpg.de/auriga/data_new.html) and [release paper](https://arxiv.org/abs/2401.08750) identify available data products and citation context.

A processed legacy file with undocumented units or frame remains scientifically unverified. Recover authoritative metadata before claiming physical interpretation; adding schema labels alone is not a conversion. LMC ancestry tags and observer orientation also need their own provenance.

## Already-processed legacy files

For a legacy root-dataset file whose coordinates are already physical kpc and whose velocities are already km/s in the intended Galactocentric rest frame, the importer requires an explicit confirmation:

```sh
python -m lmc_auriga import-legacy data/legacy.hdf5 data/processed/snapshot.hdf5 --provenance "Processed source; units and host frame confirmed by its producer" --confirm-processed-frame
```

Use the flag only with supporting knowledge of that source. The importer records the confirmation and source checksum. It copies coordinates and velocities without numerical unit conversion, recentering, bulk-velocity subtraction, or axis rotation. Under its explicit legacy tag convention, `-1` maps to host/MW and strictly positive values map to tagged/LMC; zero and other negative values are rejected. The original values are retained as `Original_LMC_tag`. Existing positive `Weights` are preserved; absent weights imply particle-number weighting.

This creates a documented processed container, not an independent audit of the original simulation. Verify the source's identity before pairing it with halo-specific circular speeds, observer orientations, or calibration data. Distinct filenames are insufficient evidence of distinct snapshots: inspect checksums and authoritative source records. Keep identity checks and private source metadata with the private input rather than in public example artifacts.

## Public fixture and private inputs

The generator writes all required fields, equal weights, its seed and parameters, dependency versions, random-generator name, and a hash of its source. It refuses to replace an existing snapshot. [Methods](methods.md#synthetic-demonstration) specify the toy model.

The repository ignores `data/`, `results/`, and HDF5 files. The committed showcase contains only reproducible synthetic summary tables and figures. Keep private inputs, identifiers, source records, and any reports containing private metadata under ignored directories; inspect report contents before sharing them.
