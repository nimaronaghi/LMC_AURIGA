"""Fixed-size scientific figures from saved analysis tables, without rerunning data."""

import csv
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = ("total", "mw", "lmc")
STYLES = {
    "total": {"color": "#222222", "linestyle": "-", "label": "Total"},
    "mw": {"color": "#0072B2", "linestyle": "--", "label": "MW contribution"},
    "lmc": {"color": "#D55E00", "linestyle": "-.", "label": "LMC contribution"},
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


def _check_additive(table, suffix):
    if not np.allclose(table[f"total_{suffix}"], table[f"mw_{suffix}"] + table[f"lmc_{suffix}"],
                       rtol=1e-9, atol=1e-15):
        raise ValueError(f"{suffix} components must add to total under the common normalization")


def make_figures(directory):
    """Render saved PDF, tail probability and mean inverse-speed measurements.

    Read ``speed_distribution.csv``, ``observables.csv`` and ``report.json``.
    Write ``scientific_review.pdf/.svg/.png`` and ``caption.txt`` in the same
    directory and return their paths. The 7.2 by 2.7 inch canvas is preserved
    in vector exports and in the 600 dpi PNG. Source measurements are read only.
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
        "(a) Phase-averaged speed probability density, shown over the recorded histogram bins. "
        "(b) Tail probability P(v > v_min). (c) Mean inverse speed eta(v_min), "
        "the population-weighted mean of 1/v for v > v_min. Threshold inequalities are strict. "
        "MW and LMC curves are contributions normalized by the same total selected population weight "
        "and phase count, so they add to the total; they are not separately unit-normalized distributions. "
        "Phase-resolved contributions are averaged after computing each observable, not by histogramming "
        "a particle's mean speed. Exact zeros are omitted from logarithmic panels (b,c); no positive floor "
        "is substituted. Lines in (b,c) connect recorded threshold evaluations, not fitted models. "
        "No confidence intervals or measurement-uncertainty estimates are shown. "
        "The figure is rendered from speed_distribution.csv, observables.csv and report.json; "
        "the input tables and analysis results are unchanged."
    )

    # Import plotting only when requested, and avoid pyplot/global backend changes.
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False, "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.65, "lines.linewidth": 1.15,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "legend.frameon": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "savefig.bbox": None, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "lmc-auriga-scientific-review",
    }
    paths = [directory / f"scientific_review.{extension}" for extension in ("pdf", "svg", "png")]
    caption_path = directory / "caption.txt"
    with mpl.rc_context(style):
        figure = Figure(figsize=(7.2, 2.7), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        figure.suptitle(heading, fontsize=9, fontweight="bold")
        edges = np.r_[left, right[-1]]
        for component in COMPONENTS:
            axes[0].stairs(distribution[f"{component}_pdf"], edges, **STYLES[component])
            for axis, quantity in zip(axes[1:], ("tail", "eta")):
                axis.plot(observables["vmin_km_s"], _positive_or_nan(observables[f"{component}_{quantity}"]),
                          **STYLES[component])
        titles = ("(a)  Speed distribution", "(b)  Speed tail", "(c)  Mean inverse speed")
        for axis, title in zip(axes, titles):
            axis.set_title(title, loc="left", fontweight="bold")
            axis.tick_params(which="both", direction="in", top=True, right=True)
            axis.margins(x=0.02)
        axes[0].set_xlabel(r"$v\;[\mathrm{km\,s^{-1}}]$")
        axes[0].set_ylabel(r"$f(v)\;[\mathrm{s\,km^{-1}}]$")
        axes[0].set_ylim(bottom=0)
        axes[0].legend(loc="best")
        for axis, quantity in zip(axes[1:], ("tail", "eta")):
            axis.set_xlabel(r"$v_{\min}\;[\mathrm{km\,s^{-1}}]$")
            axis.set_yscale("log")
            if len(observables["vmin_km_s"]) > 1:
                # Keep the measured threshold domain visible even where the
                # final values are zero and therefore absent from a log axis.
                axis.set_xlim(observables["vmin_km_s"][[0, -1]])
            if not any((observables[f"{component}_{quantity}"] > 0).any() for component in COMPONENTS):
                axis.set_ylim(1e-6, 1)
                axis.text(0.5, 0.5, "All recorded values are zero\n(no positive values to plot)",
                          transform=axis.transAxes, ha="center", va="center", fontsize=7)
        axes[1].set_ylabel(r"$P(v>v_{\min})$")
        axes[2].set_ylabel(r"$\eta(v_{\min})\;[\mathrm{s\,km^{-1}}]$")
        try:
            figure.savefig(paths[0], format="pdf", bbox_inches=None,
                           metadata={"Title": heading, "Subject": caption, "Creator": "LMC_AURIGA",
                                     "CreationDate": None, "ModDate": None})
            figure.savefig(paths[1], format="svg", bbox_inches=None,
                           metadata={"Title": heading, "Description": caption, "Creator": "LMC_AURIGA", "Date": None})
            figure.savefig(paths[2], format="png", dpi=600, bbox_inches=None,
                           metadata={"Title": heading, "Description": caption})
        finally:
            figure.clear()
    caption_path.write_text(caption + "\n", encoding="utf-8")
    return paths + [caption_path]
