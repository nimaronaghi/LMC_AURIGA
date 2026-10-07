"""Processed-snapshot schema and bounded reader checks."""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga.io import iter_snapshot, read_metadata


class TestSnapshotIO(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "snapshot.hdf5"

    def fixture(self, count=7, weights=True):
        with h5py.File(self.path, "w") as snapshot:
            snapshot.attrs.update(schema_version=1, position_unit="kpc", velocity_unit="km/s",
                                  frame="galactocentric", provenance="test fixture")
            snapshot["Coordinates"] = np.arange(count * 3, dtype=float).reshape(count, 3)
            snapshot["Velocities"] = np.ones((count, 3)) * 2
            snapshot["LMC_tag"] = np.where(np.arange(count) % 2, -1, 1)
            if weights:
                snapshot["Weights"] = np.arange(count, dtype=float) + 1

    def test_chunk_parity_and_bounded_rows(self):
        self.fixture()
        metadata = read_metadata(self.path)
        self.assertEqual(metadata["n_particles"], 7)
        self.assertTrue(metadata["has_weights"])
        chunks = list(iter_snapshot(self.path, chunk_size=3))
        self.assertEqual([len(chunk[0]) for chunk in chunks], [3, 3, 1])
        whole = next(iter_snapshot(self.path, chunk_size=100))
        for index, expected in enumerate(whole):
            np.testing.assert_array_equal(np.concatenate([chunk[index] for chunk in chunks]), expected)
        self.assertEqual(chunks[0][2].dtype, np.dtype("int8"))

    def test_missing_weights_are_ones_and_empty_snapshot_yields_nothing(self):
        self.fixture(weights=False)
        self.assertFalse(read_metadata(self.path)["has_weights"])
        for _, _, _, weights in iter_snapshot(self.path, chunk_size=2):
            np.testing.assert_array_equal(weights, np.ones(weights.size))
        self.fixture(count=0)
        self.assertEqual(read_metadata(self.path)["n_particles"], 0)
        self.assertEqual(list(iter_snapshot(self.path)), [])

    def test_rejects_missing_or_wrong_schema_metadata(self):
        for name, value in (("schema_version", 2), ("schema_version", 1.0),
                            ("position_unit", "Mpc"), ("velocity_unit", "code units"),
                            ("frame", "heliocentric"), ("provenance", "")):
            self.fixture()
            with h5py.File(self.path, "r+") as snapshot:
                snapshot.attrs[name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                read_metadata(self.path)
        self.fixture()
        with h5py.File(self.path, "r+") as snapshot:
            del snapshot.attrs["position_unit"]
        with self.assertRaisesRegex(ValueError, "position_unit"):
            list(iter_snapshot(self.path))

    def test_rejects_mismatched_shapes_and_non_numeric_fields(self):
        for name, values in (("Coordinates", np.zeros((7, 2))),
                             ("Velocities", np.zeros((6, 3))),
                             ("LMC_tag", np.ones((7, 1))), ("Weights", np.ones(6)),
                             ("Velocities", np.ones((7, 3), dtype=complex))):
            self.fixture()
            with h5py.File(self.path, "r+") as snapshot:
                del snapshot[name]
                snapshot[name] = values
            with self.subTest(name=name, shape=values.shape), self.assertRaises(ValueError):
                read_metadata(self.path)
        self.fixture()
        with h5py.File(self.path, "r+") as snapshot:
            del snapshot["Coordinates"]
        with self.assertRaisesRegex(ValueError, "Coordinates"):
            read_metadata(self.path)

    def test_late_chunk_invalid_values_are_not_silently_accepted(self):
        for name, value in (("Coordinates", np.nan), ("Velocities", np.inf),
                            ("LMC_tag", 0), ("Weights", 0), ("Weights", -1),
                            ("Weights", np.nan)):
            self.fixture()
            with h5py.File(self.path, "r+") as snapshot:
                snapshot[name][5] = value
            chunks = iter_snapshot(self.path, chunk_size=3)
            self.assertEqual(len(next(chunks)[0]), 3)
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                next(chunks)

    def test_chunk_size_validation(self):
        self.fixture()
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                list(iter_snapshot(self.path, chunk_size=size))


if __name__ == "__main__":
    unittest.main()
