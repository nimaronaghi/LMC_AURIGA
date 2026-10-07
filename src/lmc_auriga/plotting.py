"""Fixed-size scientific figures from saved analysis tables, without rerunning data."""

import csv
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = ("total", "mw", "lmc")
STYLES = {
    "total": {"color": "#111111", "linestyle": "-", "label": "MW + LMC"},
    "mw": {"color": "#c52c2c", "linestyle": "--", "label": "MW contribution"},
    "lmc": {"color": "#3157bb", "linestyle": "-.", "label": "LMC contribution"},
}


def _read_table(path, columns):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing required columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} contains no measurements")
    table = {name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
             for name in columns}
    if any(not np.isfinite(values).all() or (values < 0).any() for values in table.values()):
        raise ValueError(f"{path.name} must contain finite nonnegative measurements")
    return table


def _positive_or_nan(values):
    """Omit exact zeros from logarithmic axes without inventing small values."""
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > 0, values, np.nan)


def _safe_ratio(numerator, denominator):
    """Leave ratios undefined where the empirical denominator is zero."""
    numerator, denominator = np.broadcast_arrays(
        np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64))
    return np.divide(numerator, denominator, out=np.full(numerator.shape, np.nan), where=denominator > 0)


def _check_additive(table, suffix):
    if not np.allclose(table[f"total_{suffix}"], table[f"mw_{suffix}"] + table[f"lmc_{suffix}"],
                       rtol=1e-9, atol=1e-15):
        raise ValueError(f"{suffix} components must add to total under the common normalization")


def make_figures(directory):
    """Render saved PDF, tail probability and mean inverse-speed measurements.

    Read ``speed_distribution.csv``, ``observables.csv`` and ``report.json``.
    Write a two-column main figure with comparison panels (7.2 by 4.4 inches),
    a supplementary speed-tail figure (3.5 by 3.2 inches), and ``caption.txt``.
    Each figure is exported as PDF/SVG/600 dpi PNG. Source tables are read only.
    Layout is informed by figures 4 and 6 of Smith-Orlik et al. (2023), while
    preserving this package's observer frame and additive weight normalization.
    """
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    distribution = _read_table(directory / "speed_distribution.csv",
                               ["speed_left_km_s", "speed_right_km_s"] +
                               [f"{component}_pdf" for component in COMPONENTS])
    observables = _read_table(directory / "observables.csv", ["vmin_km_s"] +
                              [f"{component}_{quantity}" for quantity in ("tail", "eta") for component in COMPONENTS])
    left, right = distribution["speed_left_km_s"], distribution["speed_right_km_s"]
    if (right <= left).any() or not np.array_equal(left[1:], right[:-1]):
        raise ValueError("speed bins must have positive widths and contiguous increasing edges")
    if (np.diff(observables["vmin_km_s"]) <= 0).any():
        raise ValueError("vmin_km_s must be strictly increasing")
    _check_additive(distribution, "pdf")
    for quantity in ("tail", "eta"):
        _check_additive(observables, quantity)
    if (observables["total_tail"] > 1 + 1e-10).any():
        raise ValueError("total tail probabilities cannot exceed one")

    configuration = report["configuration"]
    shell = configuration["shell"]
    phases = configuration["phase_count"]
    selected = report["selected_count"]
    if (len(shell) != 2 or not all(math.isfinite(value) for value in shell)
            or not 0 <= shell[0] <= shell[1]):
        raise ValueError("report shell must contain nondecreasing nonnegative radii")
    if isinstance(phases, bool) or not isinstance(phases, int) or phases < 1:
        raise ValueError("report phase_count must be a positive integer")
    if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
        raise ValueError("report selected_count must be a nonnegative integer")
    provenance = report["input_metadata"]["provenance"]
    synthetic = provenance == "synthetic"
    heading = ("SYNTHETIC DEMONSTRATION" if synthetic else "PROCESSED PARTICLE DATA")
    heading += "  |  Observer-frame speed observables"
    circular_speed = configuration.get("vc_km_s")
    circular_speed_text = f"{circular_speed:g} km/s" if circular_speed is not None else "not recorded"
    basis_rows = configuration.get("basis_rows")
    basis_text = json.dumps(basis_rows, allow_nan=False) if basis_rows is not None else "not recorded"
    if "cone_axis" not in configuration:
        selection_text = "Angular selection: not recorded. "
    elif configuration["cone_axis"] is None:
        selection_text = "Angular selection: no cone (full solid angle within the radial shell). "
    else:
        axis_text = json.dumps(configuration["cone_axis"], allow_nan=False)
        half_angle = configuration.get("half_angle_deg")
        angle_text = f"{half_angle:g} degrees" if half_angle is not None else "not recorded"
        selection_text = f"Cone axis in simulation coordinates: {axis_text}; half-angle: {angle_text}. "
    caption = (
        ("Synthetic demonstration; these measurements are not observational or Auriga simulation results. "
         if synthetic else f"Processed particle data; declared provenance: {provenance}. ") +
        f"Selected particle count: {selected}; galactocentric shell: {shell[0]:g}–{shell[1]:g} kpc; "
        f"observer phases: {phases}. "
        f"Local circular speed vc: {circular_speed_text}. "
        f"{selection_text}"
        f"Basis rows (local Galactic U,V,W axes in simulation coordinates): {basis_text}. "
        "Main figure: (a) phase-averaged observer-frame speed density; "
        "(b) mean inverse speed eta(v_min), "
        "the population-weighted mean of 1/v for v > v_min. Threshold inequalities are strict. "
        "The vertical scales of both upper panels are expressed in units of 10^-3 s/km. "
        "Lower panels show f_total/f_MW and (eta_total - eta_MW)/eta_MW, respectively, "
        "using the same weighted component contributions as the upper panels. "
        "Zero-denominator ratios are undefined and omitted, without a denominator floor. "
        "Supplementary figure: strict tail probability P(v > v_min). "
        "MW and LMC curves are contributions normalized by the same total selected population weight "
        "and phase count, so they add to the total; they are not separately unit-normalized distributions. "
        "Phase-resolved contributions are averaged after computing each observable, not by histogramming "
        "a particle's mean speed. Exact zeros are omitted from logarithmic panels; no positive floor "
        "is substituted. Threshold curves connect recorded evaluations, not fitted models. "
        "No confidence intervals or measurement-uncertainty estimates are shown. "
        "Layout and black/red/blue population encoding are informed by figures 4 and 6 of "
        "Smith-Orlik et al. (2023), https://arxiv.org/abs/2302.04281; the underlying data, "
        "normalization and frame are those stated here, not a reproduction of that paper. "
        "The figure is rendered from speed_distribution.csv, observables.csv and report.json; "
        "the input tables and analysis results are unchanged."
    )

    # Import plotting only when requested, and avoid pyplot/global backend changes.
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    style = {
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "stix", "text.usetex": False, "font.size": 9,
        "axes.labelsize": 10, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.65, "lines.linewidth": 1.1,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "legend.frameon": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "savefig.bbox": None, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "lmc-auriga-scientific-review",
    }
    paths = []
    caption_path = directory / "caption.txt"
    with mpl.rc_context(style):
        figure = Figure(figsize=(7.2, 4.4))
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(2, 2, height_ratios=(3.2, 1),
                                  left=.105, right=.96, bottom=.13, top=.96, hspace=.06, wspace=.36)
        speed_axis = figure.add_subplot(grid[0, 0])
        eta_axis = figure.add_subplot(grid[0, 1])
        speed_ratio_axis = figure.add_subplot(grid[1, 0], sharex=speed_axis)
        eta_ratio_axis = figure.add_subplot(grid[1, 1], sharex=eta_axis)
        axes = [speed_axis, eta_axis, speed_ratio_axis, eta_ratio_axis]
        edges = np.r_[left, right[-1]]
        thresholds = observables["vmin_km_s"]
        for component in COMPONENTS:
            speed_axis.stairs(1000 * distribution[f"{component}_pdf"], edges, **STYLES[component])
            eta_axis.plot(thresholds, 1000 * _positive_or_nan(observables[f"{component}_eta"]),
                          **STYLES[component])
        speed_ratio = _safe_ratio(distribution["total_pdf"], distribution["mw_pdf"])
        eta_ratio = _safe_ratio(observables["total_eta"] - observables["mw_eta"], observables["mw_eta"])
        speed_ratio_axis.stairs(speed_ratio, edges, baseline=None, color="#111111", linewidth=.9)
        eta_ratio_axis.plot(thresholds, eta_ratio, color="#111111", linewidth=.9)
        for axis, reference in ((speed_ratio_axis, 1), (eta_ratio_axis, 0)):
            axis.axhline(reference, color=".55", linewidth=.6, linestyle=":")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
            axis.set_ylim(bottom=0)
        speed_ratio_axis.set_ylim(bottom=.95)
        for axis, ratio in ((speed_ratio_axis, speed_ratio), (eta_ratio_axis, eta_ratio)):
            if not np.isfinite(ratio).any():
                axis.text(.5, .5, "Undefined: no MW support", transform=axis.transAxes,
                          ha="center", va="center", fontsize=7)
        for axis in axes:
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.grid(which="major", color=".88", linewidth=.35)
            axis.set_axisbelow(True)
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        speed_axis.set_xlim(edges[0], edges[-1])
        eta_axis.set_xlim(thresholds[0], thresholds[-1] if len(thresholds) > 1 else thresholds[0] + 1)
        speed_axis.set_ylim(bottom=0, top=max(float(distribution["total_pdf"].max()) * 1300, 1))
        speed_axis.tick_params(labelbottom=False)
        eta_axis.tick_params(labelbottom=False)
        eta_axis.set_yscale("log")
        _log_limits(eta_axis, np.concatenate([1000 * observables[f"{c}_eta"] for c in COMPONENTS]))
        speed_axis.set_ylabel(r"$f(v)\;[10^{-3}\,\mathrm{s\,km^{-1}}]$")
        eta_axis.set_ylabel(r"$\eta(v_{\min})\;[10^{-3}\,\mathrm{s\,km^{-1}}]$")
        speed_ratio_axis.set_ylabel(r"$f_{\rm tot}/f_{\rm MW}$", fontsize=9)
        eta_ratio_axis.set_ylabel(r"$\Delta\eta/\eta_{\rm MW}$", fontsize=9)
        speed_ratio_axis.set_xlabel(r"$v\;[\mathrm{km\,s^{-1}}]$")
        eta_ratio_axis.set_xlabel(r"$v_{\min}\;[\mathrm{km\,s^{-1}}]$")
        sample_label = "Synthetic fixture" if synthetic else "Processed sample"
        for axis, letter in ((speed_axis, "a"), (eta_axis, "b")):
            axis.text(.04, .98, f"({letter})", transform=axis.transAxes, va="top", fontsize=9)
            axis.text(.96, .96, sample_label, transform=axis.transAxes, va="top", ha="right", color=".4", fontsize=8)
        speed_axis.text(.96, .85, rf"$N={selected:,}$", transform=speed_axis.transAxes, ha="right", va="top", fontsize=8)
        speed_axis.legend(loc="upper right", bbox_to_anchor=(1, .78), handlelength=2.3, labelspacing=.4)
        eta_axis.text(.04, .05, f"{phases} annual phases", transform=eta_axis.transAxes, color=".4", fontsize=8)
        _export_figure(figure, directory, "scientific_review", heading, caption, paths)

        tail_figure = Figure(figsize=(3.5, 3.2))
        FigureCanvasAgg(tail_figure)
        tail_axis = tail_figure.add_subplot(111)
        tail_figure.subplots_adjust(left=.19, right=.94, bottom=.17, top=.96)
        for component in COMPONENTS:
            tail_axis.plot(thresholds, _positive_or_nan(observables[f"{component}_tail"]), **STYLES[component])
        tail_axis.set_yscale("log")
        _log_limits(tail_axis, np.concatenate([observables[f"{c}_tail"] for c in COMPONENTS]))
        tail_axis.set_xlim(thresholds[0], thresholds[-1] if len(thresholds) > 1 else thresholds[0] + 1)
        tail_axis.set_xlabel(r"$v_{\min}\;[\mathrm{km\,s^{-1}}]$")
        tail_axis.set_ylabel(r"$P(v>v_{\min})$")
        tail_axis.grid(which="major", color=".88", linewidth=.35)
        tail_axis.set_axisbelow(True)
        tail_axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        tail_axis.text(.96, .96, sample_label, transform=tail_axis.transAxes, va="top", ha="right", color=".4", fontsize=8)
        tail_axis.legend(loc="lower left", fontsize=7)
        _export_figure(tail_figure, directory, "speed_tail", heading + " | Supplementary tail", caption, paths)
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return paths + [caption_path]


def _log_limits(axis, values):
    positive = np.asarray(values)[np.asarray(values) > 0]
    if positive.size:
        axis.set_ylim(10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max()) + .12))
    else:
        axis.set_ylim(1e-6, 1)
        axis.text(.5, .5, "No positive values", transform=axis.transAxes, ha="center", va="center", fontsize=8)


def _export_figure(figure, directory, stem, title, caption, paths):
    """Export identical physical canvases and embed the scientific caption."""
    try:
        for extension in ("pdf", "svg", "png"):
            path = directory / f"{stem}.{extension}"
            metadata = {"Title": title}
            if extension == "pdf":
                metadata.update(Subject=caption, Creator="LMC_AURIGA", CreationDate=None, ModDate=None)
            elif extension == "svg":
                metadata.update(Description=caption, Creator="LMC_AURIGA", Date=None)
            else:
                metadata.update(Description=caption)
            figure.savefig(path, format=extension, dpi=600, bbox_inches=None, metadata=metadata)
            paths.append(path)
    finally:
        figure.clear()
