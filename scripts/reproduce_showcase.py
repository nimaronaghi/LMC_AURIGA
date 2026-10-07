"""Regenerate the committed, explicitly synthetic figure and its numerical tables.

Run from any directory after installing the project with its plot extra.
This deliberately replaces only the generated files in examples/showcase.
It never copies an HDF5 input or imports private source data.
"""

from pathlib import Path
import shutil
import tempfile

from lmc_auriga.analysis import analyze_snapshot
from lmc_auriga.plotting import make_figures
from lmc_auriga.synthetic import write_synthetic


def main():
    root = Path(__file__).resolve().parents[1]
    work = root / "results"
    work.mkdir(exist_ok=True)
    destination = root / "examples" / "showcase"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work, prefix="showcase-") as temporary:
        output = Path(temporary)
        snapshot = output / "synthetic.hdf5"
        write_synthetic(snapshot, n_particles=20000, seed=42)
        analyze_snapshot(snapshot, output)
        make_figures(output)
        for name in ("speed_distribution.csv", "observables.csv", "anisotropy.csv", "report.json",
                     "scientific_review.pdf", "scientific_review.svg", "scientific_review.png", "caption.txt"):
            shutil.copy2(output / name, destination / name)
    print(destination)


if __name__ == "__main__":
    main()
