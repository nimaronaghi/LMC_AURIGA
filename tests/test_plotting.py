"""Optional scientific-figure integrity checks using a known weighted sample."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga.plotting import make_figures, _positive_or_nan


HAS_PLOTTING = all(importlib.util.find_spec(name) is not None for name in ("matplotlib", "PIL"))


def write_fixture(directory, provenance="synthetic"):
    """Ten unit-weight particles with speeds 50,150,250 and MW counts 1,3,1."""
    root = Path(directory)
    report = {"input_metadata": {"provenance": provenance}, "selected_count": 10,
              "configuration": {"shell": [6, 10], "phase_count": 1}}
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with (root / "speed_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["speed_left_km_s", "speed_right_km_s", "total_pdf", "mw_pdf", "lmc_pdf"])
        writer.writerows([[0, 100, .002, .001, .001], [100, 200, .005, .003, .002],
                          [200, 300, .003, .001, .002]])
    with (root / "observables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["vmin_km_s", "total_tail", "mw_tail", "lmc_tail", "total_eta", "mw_eta", "lmc_eta"])
        for threshold in (0, 100, 200, 300):
            speeds = np.array([50, 150, 250])
            mask = speeds > threshold
            mw, lmc = np.array([.1, .3, .1]), np.array([.1, .2, .2])
            mw_tail, lmc_tail = mw[mask].sum(), lmc[mask].sum()
            mw_eta, lmc_eta = (mw[mask] / speeds[mask]).sum(), (lmc[mask] / speeds[mask]).sum()
            writer.writerow([threshold, mw_tail + lmc_tail, mw_tail, lmc_tail,
                             mw_eta + lmc_eta, mw_eta, lmc_eta])
    (root / "anisotropy.csv").write_text("component,n,weight,beta\ntotal,10,10,0\nmw,5,5,-0.2\nlmc,5,5,\n",
                                        encoding="utf-8")
    return [root / name for name in ("report.json", "speed_distribution.csv", "observables.csv", "anisotropy.csv")]


@unittest.skipUnless(HAS_PLOTTING, "optional Matplotlib/Pillow dependencies are not installed")
class ScientificPlotTests(unittest.TestCase):
    def test_equal_radius_shell_accepted_by_analysis_can_be_rendered(self):
        from matplotlib.figure import Figure

        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory)
            report_path = Path(directory) / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["configuration"]["shell"] = [8, 8]
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(Figure, "savefig") as export:
                make_figures(directory)
            self.assertEqual(export.call_count, 3)
            caption = (Path(directory) / "caption.txt").read_text(encoding="utf-8")
            self.assertIn("8–8 kpc", caption)

    def test_render_preserves_measurements_and_exact_canvas_dimensions(self):
        import matplotlib as mpl
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            sources = write_fixture(directory)
            before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
            style = dict(mpl.rcParams)
            with patch("matplotlib.use", side_effect=AssertionError("global backend change")):
                outputs = make_figures(directory)
            self.assertEqual(dict(mpl.rcParams), style)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, before)
            root = Path(directory)
            with Image.open(root / "scientific_review.png") as image:
                self.assertEqual(image.size, (4320, 1620))
                for dpi in image.info["dpi"]:
                    self.assertAlmostEqual(dpi, 600, delta=.02)
                self.assertIn("SYNTHETIC DEMONSTRATION", image.info["Title"])
            svg = ET.parse(root / "scientific_review.svg").getroot()
            self.assertAlmostEqual(float(svg.attrib["width"].removesuffix("pt")), 518.4, places=5)
            self.assertAlmostEqual(float(svg.attrib["height"].removesuffix("pt")), 194.4, places=5)
            box = re.search(rb"/MediaBox\s*\[\s*([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s+([\d.+-]+)\s*\]",
                            (root / "scientific_review.pdf").read_bytes())
            self.assertIsNotNone(box)
            x0, y0, x1, y1 = map(float, box.groups())
            self.assertAlmostEqual(x1 - x0, 518.4, places=5)
            self.assertAlmostEqual(y1 - y0, 194.4, places=5)
            caption = (root / "caption.txt").read_text(encoding="utf-8")
            self.assertIn("not observational or Auriga simulation results", caption)
            self.assertIn("same total selected population weight", caption)
            self.assertIn("Exact zeros are omitted", caption)

    def test_logarithmic_panels_preserve_values_and_processed_data_label(self):
        from matplotlib.figure import Figure

        snapshots = []

        def capture(figure, *args, **kwargs):
            snapshots.append({"heading": figure._suptitle.get_text(),
                              "scales": [axis.get_yscale() for axis in figure.axes],
                              "threshold_domain": figure.axes[1].get_xlim(),
                              "tail": [line.get_ydata().copy() for line in figure.axes[1].lines]})

        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, "processed test fixture")
            with patch.object(Figure, "savefig", autospec=True, side_effect=capture):
                make_figures(directory)
            caption = (Path(directory) / "caption.txt").read_text(encoding="utf-8")
        self.assertIn("PROCESSED PARTICLE DATA", snapshots[0]["heading"])
        self.assertEqual(snapshots[0]["scales"], ["linear", "log", "log"])
        self.assertEqual(snapshots[0]["threshold_domain"], (0, 300))
        np.testing.assert_allclose(snapshots[0]["tail"][0][:3], [1, .8, .3])
        self.assertTrue(all(np.isnan(values[-1]) for values in snapshots[0]["tail"]))
        np.testing.assert_array_equal(_positive_or_nan([1e-20, 1]), [1e-20, 1])
        self.assertTrue(np.isnan(_positive_or_nan([0])[0]))
        self.assertIn("declared provenance: processed test fixture", caption)
        self.assertIn("Local circular speed vc: not recorded", caption)
        self.assertIn("Angular selection: not recorded", caption)
        self.assertIn("Basis rows (local Galactic U,V,W axes in simulation coordinates): not recorded", caption)

    def test_caption_records_actual_observer_orientation_and_angular_selection(self):
        from matplotlib.figure import Figure

        basis = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory, "user-confirmed processed sample")
            path = Path(directory) / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["configuration"].update(vc_km_s=237.5, basis_rows=basis, cone_axis=None)
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(Figure, "savefig"):
                make_figures(directory)
            caption = (Path(directory) / "caption.txt").read_text(encoding="utf-8")
            self.assertIn("Local circular speed vc: 237.5 km/s", caption)
            self.assertIn("no cone (full solid angle within the radial shell)", caption)
            self.assertIn(json.dumps(basis), caption)

            report["configuration"].update(cone_axis=[1.0, 0.0, 0.0], half_angle_deg=30.0)
            path.write_text(json.dumps(report), encoding="utf-8")
            with patch.object(Figure, "savefig"):
                make_figures(directory)
            caption = (Path(directory) / "caption.txt").read_text(encoding="utf-8")
            self.assertIn("Cone axis in simulation coordinates: [1.0, 0.0, 0.0]; half-angle: 30 degrees", caption)
            self.assertNotIn("no cone", caption)

    def test_inconsistent_component_normalization_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory)
            path = Path(directory) / "speed_distribution.csv"
            contents = path.read_text(encoding="utf-8")
            path.write_text(contents.replace("0.002,0.001,0.001", "0.009,0.001,0.001"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "add to total"):
                make_figures(directory)
            self.assertFalse((Path(directory) / "scientific_review.png").exists())

    def test_style_is_restored_after_export_failure(self):
        import matplotlib as mpl
        from matplotlib.figure import Figure

        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory)
            with mpl.rc_context({"font.size": 13, "axes.linewidth": 2, "savefig.bbox": "tight"}):
                before = dict(mpl.rcParams)
                with patch.object(Figure, "savefig", side_effect=OSError("simulated failure")):
                    with self.assertRaisesRegex(OSError, "simulated failure"):
                        make_figures(directory)
                self.assertEqual(dict(mpl.rcParams), before)


if __name__ == "__main__":
    unittest.main()
