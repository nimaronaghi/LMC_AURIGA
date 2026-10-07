"""Reproducibility and known statistical properties of the toy fixture."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga.io import iter_snapshot, read_metadata
from lmc_auriga.synthetic import write_synthetic


class TestSyntheticSnapshot(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.hdf5"

    def test_seed_reproduces_arrays_and_explicit_metadata(self):
        metadata = write_synthetic(self.path, n_particles=101, seed=27, lmc_fraction=0.05)
        second = Path(self.directory.name) / "second.hdf5"
        repeated_metadata = write_synthetic(second, n_particles=101, seed=27, lmc_fraction=0.05)
        self.assertEqual(metadata, repeated_metadata)
        self.assertEqual(metadata["provenance"], "synthetic")
        self.assertEqual(metadata["seed"], 27)
        self.assertEqual(metadata["lmc_count"], 5)
        self.assertEqual(metadata["mw_count"], 96)
        self.assertEqual(len(metadata["generator_source_sha256"]), 64)
        self.assertIn("not a gravitational simulation", metadata["model_scope"])
        self.assertEqual(json.loads(metadata["parameters_json"])["radial_shell_kpc"], [4, 14])
        with h5py.File(self.path, "r") as first, h5py.File(second, "r") as other:
            for name in ("Coordinates", "Velocities", "LMC_tag", "Weights"):
                np.testing.assert_array_equal(first[name][:], other[name][:])
        # The returned metadata is JSON serializable without a special encoder.
        json.dumps(read_metadata(self.path), allow_nan=False)

    def test_different_seeds_change_particle_realization(self):
        write_synthetic(self.path, n_particles=20, seed=1)
        other = Path(self.directory.name) / "other.hdf5"
        write_synthetic(other, n_particles=20, seed=2)
        with h5py.File(self.path, "r") as first, h5py.File(other, "r") as second:
            self.assertFalse(np.array_equal(first["Coordinates"][:], second["Coordinates"][:]))

    def test_position_and_velocity_statistics_match_declared_toy_model(self):
        count = 20000
        write_synthetic(self.path, n_particles=count, seed=42, lmc_fraction=0.2)
        coordinates, velocities, tags, weights = next(iter_snapshot(self.path))
        radius = np.linalg.norm(coordinates, axis=1)
        self.assertTrue(((radius >= 4) & (radius <= 14)).all())
        volume_fraction = (radius**3 - 4**3) / (14**3 - 4**3)
        self.assertAlmostEqual(float(volume_fraction.mean()), 0.5, delta=0.01)
        directions = coordinates / radius[:, None]
        np.testing.assert_allclose(directions.mean(axis=0), 0, atol=0.015)
        np.testing.assert_allclose((directions**2).mean(axis=0), 1 / 3, atol=0.012)
        self.assertEqual(int((tags == 1).sum()), 4000)
        np.testing.assert_array_equal(weights, np.ones(count))
        np.testing.assert_allclose(velocities[tags == -1].mean(axis=0), [0, 0, 0], atol=4)
        np.testing.assert_allclose(velocities[tags == -1].std(axis=0), [155, 155, 155], atol=4)
        np.testing.assert_allclose(velocities[tags == 1].mean(axis=0), [0, -300, 80], atol=4)
        np.testing.assert_allclose(velocities[tags == 1].std(axis=0), [70, 50, 50], atol=4)

    def test_empty_and_single_component_fixtures(self):
        metadata = write_synthetic(self.path, n_particles=0)
        self.assertEqual(metadata["n_particles"], 0)
        self.assertEqual(list(iter_snapshot(self.path)), [])
        for fraction, expected_tag in ((0, -1), (1, 1)):
            path = Path(self.directory.name) / f"component-{fraction}.hdf5"
            write_synthetic(path, n_particles=5, lmc_fraction=fraction)
            self.assertTrue((next(iter_snapshot(path))[2] == expected_tag).all())

    def test_refuses_existing_destination_and_invalid_parameters(self):
        self.path.write_bytes(b"existing user data")
        with self.assertRaises(FileExistsError):
            write_synthetic(self.path, n_particles=10)
        self.assertEqual(self.path.read_bytes(), b"existing user data")
        other = Path(self.directory.name) / "invalid.hdf5"
        for kwargs in ({"n_particles": -1}, {"n_particles": True}, {"n_particles": 1.5},
                       {"seed": -1}, {"seed": True}, {"lmc_fraction": -0.1},
                       {"lmc_fraction": 1.1}, {"lmc_fraction": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                write_synthetic(other, **kwargs)
        self.assertFalse(other.exists())


if __name__ == "__main__":
    unittest.main()
