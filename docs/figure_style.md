# Figure presentation

The visual reference is **Smith-Orlik et al. (2023), figures 4 and 6** in
[*The impact of the Large Magellanic Cloud on dark matter direct detection signals*](https://arxiv.org/abs/2302.04281).
These figures use black/red/blue population coding and a short comparison strip
below each principal panel. We adapt that hierarchy to the observables available
here. The figures do not reproduce the paper's data or its statistical analysis.

## Main figure

`scientific_review` uses a 7.2 x 4.4 inch canvas. The left column shows the
observer-frame speed density and its total/MW-contribution ratio. The right
column shows the annual mean inverse speed and the relative added contribution
`(eta_total - eta_MW) / eta_MW`. Both upper axes use units of `10^-3 s/km`;
the numerical table values are multiplied by 1,000 for this display only.
The lower panels share the corresponding upper horizontal axis.

All population curves retain the analysis's **common total selected-weight
normalization**. MW and LMC contributions add to the total. In particular, the
MW denominator in the comparison strips is its weighted contribution, not a
separately unit-normalized MW-only distribution. Ratios are omitted wherever
the empirical MW denominator is zero, without adding a small denominator.
The LMC contribution is not independently amplified or rescaled.

The total is black and solid, the MW contribution red and dashed, and the LMC
contribution blue and dash-dotted. Line styles preserve identification in
grayscale. Typography is compact and regular-weight, with mathematical labels,
thin boxed axes, inward ticks, a light major grid, and small in-panel annotations.
There is no banner title. Synthetic inputs retain a visible `Synthetic fixture`
annotation, and full provenance and model choices remain in the caption.

## Supplement and export

`speed_tail` is a 3.5 x 3.2 inch single-column figure of the strict tail probability.
It uses the same population coding and a logarithmic vertical axis. Exact zero
values are absent from logarithmic plots, without a positive floor. All saved
thresholds remain within the horizontal domain.

Each figure exports to vector PDF and SVG and to 600 dpi PNG, without cropping
its physical canvas. The figures are generated from the saved tables; no
smoothing, fitted curves, confidence bands, or extra measurements are introduced.
The caption explains that the public input is a synthetic kinematic fixture.

Run `python scripts/reproduce_showcase.py` to rebuild the committed synthetic
example, or `python -m lmc_auriga figures RESULTS_DIRECTORY` to restyle saved
results. The latter reads numerical tables without changing them.
