# Scientific audit: 7 October 2026

## Assessment and correction

The observer transformation, phase averaging, weighted speed density, strict inverse-speed sum, and additive population normalization are consistent with the documented kinematic definitions. A numerical edge case was found and corrected: an exactly constant radial velocity could acquire a tiny roundoff variance and produce an enormous, meaningless anisotropy. The streaming moments now use a fixed observed velocity reference across chunks. Exact zero radial dispersion stays zero and gives an undefined beta; genuinely small nonzero dispersion is retained.

Regression tests cover decimal constants, unequal weights, chunk partitions, small analytical dispersions, and complete HDF5 processing. The full local suite after the fix passed 132 tests and 79 subtests. These validate the implementation, not the identity or scientific interpretation of external data.

## Scientific checks

- The observer velocity is subtracted, using the declared right-handed Galactic basis. Orbital coefficients and phase signs match [Savage, Freese & Gondolo, Eqs. 13-15](https://arxiv.org/pdf/astro-ph/0607121). The circular orbit is an approximation, not a precision ephemeris.
- Each phase's observables are computed before averaging. Repeated phases reuse the same particles and are not independent statistical samples.
- MW and LMC contributions share the selected total weight and therefore add to the total. Speed overflow remains in the denominator and is reported; inverse-speed integrals use particles rather than histogram approximations.
- Anisotropy uses centered, host-frame spherical velocities before the observer boost. A pooled shell estimate does not establish equilibrium or a resolved beta(r) profile.
- The strict threshold and zero-speed conventions, unit conversions, component tags, and weights are documented and tested. Unit/frame declarations require independently verified preprocessing.

## Relation to the LMC literature

[Smith-Orlik et al. (2023)](https://arxiv.org/html/2302.04281v2) motivate the high-speed-tail application. This repository's showcase does not reproduce that study: it uses a synthetic full-shell fixture, observer-frame quantities, and commonly normalized component contributions. The paper's selected solar geometry, ancestry definitions, separate Galactic-frame component normalizations in Figure 4, and circular-speed rescaling are not interchangeable with these choices. Positive imported tags alone do not establish the paper's LMC ancestry selection.

[Folsom et al. (2025; revised arXiv version April 2026)](https://arxiv.org/html/2505.07924v2) analyze a large TNG50 halo sample and discuss the importance of phase-space scaling and halo-to-halo variation. Their LMC-related tail differences are interpreted within that ensemble. A toy tagged stream or a single undocumented snapshot cannot establish a population-wide LMC effect. A change of observer coordinates must also not be confused with a physical rescaling of a simulated galaxy.

[Herrera (2026)](https://arxiv.org/html/2601.05332v1) illustrates that velocity sensitivity depends on interaction and threshold assumptions. Mean inverse speed is a useful ingredient for standard scattering calculations, not a complete description of every interaction. Density, particle/nuclear physics, detector response, and inference are absent here.

## Figures, data, and reproducibility

Both showcase PDFs were rendered and inspected at print scale: labels, numbers, legends, ratio strips, and annotations are legible without overlaps or clipping. PDF/SVG canvases and 600 dpi PNG dimensions agree. Captions specify frame, selection, common normalization, and the absence of uncertainty intervals; undefined ratios and log-scale zeros are omitted rather than replaced by invented floors.

The tracked showcase is synthetic. Restricted particle files and derived real-data outputs remain outside the tracked tree. A real application additionally needs unique halo/snapshot provenance, validated ancestry, observer geometry, selection sensitivity, phase convergence, and appropriate sampling/physical uncertainty estimates. Duplicate input files cannot be treated as independent halos.

```bash
python -m pytest -q
python scripts/reproduce_showcase.py
```

The regenerated report records the current numerical source hashes. Historical kernel timing tables retain their original benchmark metadata; the anisotropy correction does not change the timed observer-speed kernel or establish a new performance result.
