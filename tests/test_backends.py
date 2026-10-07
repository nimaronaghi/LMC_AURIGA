"""Analytic and cross-backend checks for phase-resolved observer speeds."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga.backends import shifted_speeds
from lmc_auriga import backends


class SpeedKernelTests(unittest.TestCase):
    def test_missing_numba_is_a_cli_catchable_import_error_with_correct_extra(self):
        with patch.object(backends, "_COMPILED_NUMBA_KERNEL", None):
            with patch.dict(sys.modules, {"numba": None}):
                with self.assertRaisesRegex(ImportError, r"parallel.*pip install.*\[parallel\]"):
                    shifted_speeds([[1, 2, 3]], [[0, 0, 0]], backend="numba")

    def test_analytic_speeds_preserve_every_phase_and_particle(self):
        velocities = np.array([[3, 4, 0], [0, 0, 0], [-3, -4, 0]])
        observers = np.array([[0, 0, 0], [3, 4, 0]])
        expected = [[5, 0, 5], [0, 5, 10]]
        for backend in ("python", "numpy"):
            with self.subTest(backend=backend):
                actual = shifted_speeds(velocities, observers, backend)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.shape, (2, 3))
                self.assertEqual(actual.dtype, np.float64)

    def test_phase_distribution_is_not_replaced_by_mean_particle_speed(self):
        speeds = shifted_speeds([[0, 0, 0]], [[0, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(speeds[:, 0], [0, 2])
        phase_counts, _ = np.histogram(speeds.ravel(), bins=[0, 0.5, 1.5, 2.5])
        mean_counts, _ = np.histogram(speeds.mean(axis=0), bins=[0, 0.5, 1.5, 2.5])
        np.testing.assert_array_equal(phase_counts, [1, 0, 1])
        np.testing.assert_array_equal(mean_counts, [0, 1, 0])

    def test_numpy_matches_scalar_reference_and_shared_frame_transform(self):
        rng = np.random.default_rng(73)
        velocities = rng.normal(size=(37, 3)) * 200
        observers = rng.normal(size=(7, 3)) * 20
        reference = shifted_speeds(velocities, observers, "python")
        actual = shifted_speeds(velocities, observers, "numpy")
        np.testing.assert_allclose(actual, reference, rtol=2e-15, atol=1e-12)
        rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        boost = np.array([31, -42, 15])
        transformed = shifted_speeds((velocities + boost) @ rotation,
                                     (observers + boost) @ rotation)
        np.testing.assert_allclose(transformed, reference, rtol=2e-15, atol=1e-12)

    def test_conversion_handles_noncontiguous_inputs_without_mutation(self):
        velocities = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        observers = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        before_velocities, before_observers = velocities.copy(), observers.copy()
        actual = shifted_speeds(velocities, observers)
        expected = shifted_speeds(before_velocities, before_observers, "python")
        np.testing.assert_allclose(actual, expected, rtol=2e-15)
        np.testing.assert_array_equal(velocities, before_velocities)
        np.testing.assert_array_equal(observers, before_observers)

    def test_empty_selection_has_explicit_two_dimensional_output(self):
        for backend in ("python", "numpy"):
            with self.subTest(backend=backend):
                result = shifted_speeds(np.empty((0, 3)), np.zeros((4, 3)), backend)
                self.assertEqual(result.shape, (4, 0))
                self.assertEqual(result.dtype, np.float64)

    def test_rejects_invalid_shapes_values_and_backend_names(self):
        valid = np.ones((2, 3))
        invalid = [np.ones(3), np.ones((2, 4)), [[1, 2, np.nan]], [[1, 2, np.inf]],
                   [["1", "2", "3"]], [[True, False, True]], [[1j, 2, 3]]]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    shifted_speeds(value, valid)
                with self.assertRaises(ValueError):
                    shifted_speeds(valid, value)
        with self.assertRaises(ValueError):
            shifted_speeds(valid, np.empty((0, 3)))
        with self.assertRaises(ValueError):
            shifted_speeds(valid, valid, "cuda")

    @unittest.skipUnless(importlib.util.find_spec("numba") is not None, "optional Numba is not installed")
    def test_numba_parallel_output_is_ordered_and_matches_reference(self):
        rng = np.random.default_rng(73)
        velocities, observers = rng.normal(size=(513, 3)), rng.normal(size=(5, 3))
        expected = shifted_speeds(velocities, observers, "python")
        for _ in range(3):
            actual = shifted_speeds(velocities, observers, "numba")
            np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-14)
        self.assertEqual(shifted_speeds(np.empty((0, 3)), observers, "numba").shape, (5, 0))


if __name__ == "__main__":
    unittest.main()
