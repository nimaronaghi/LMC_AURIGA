"""Benchmark scope and result-integrity checks, without speedup assertions."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga import backends
from lmc_auriga.benchmark import run_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_existing_measurements_are_preserved_before_kernel_initialization(self):
        for filename in ("raw_timings.csv", "metadata.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                existing = output / filename
                existing.write_bytes(b"original measurement evidence")
                with patch.object(backends, "_get_kernel", side_effect=AssertionError("kernel initialized")):
                    with self.assertRaisesRegex(FileExistsError, "fresh directory"):
                        run_benchmark(output, counts=[3], repeats=1, backends=("numpy",))
                self.assertEqual(existing.read_bytes(), b"original measurement evidence")
                self.assertEqual([path.name for path in output.iterdir()], [filename])

    def test_backends_receive_identical_inputs_and_warmup_is_not_a_sample(self):
        calls, snapshots = [], {}
        original_get_kernel = backends._get_kernel

        def recording_kernel(name):
            original = original_get_kernel(name)

            def evaluate(velocities, observers):
                calls.append(name)
                snapshot = velocities.tobytes(), observers.tobytes()
                snapshots.setdefault(name, snapshot)
                self.assertEqual(snapshot, snapshots[name])
                result = original(velocities, observers)
                self.assertEqual((velocities.tobytes(), observers.tobytes()), snapshot)
                return result

            return evaluate

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(backends, "_get_kernel", side_effect=recording_kernel):
                result = run_benchmark(directory, counts=[9], repeats=2, seed=19, backends=("numpy", "python"))
        self.assertEqual(calls, ["numpy", "python", "numpy", "python", "python", "numpy"])
        self.assertEqual(snapshots["numpy"], snapshots["python"])
        self.assertEqual(len(result["samples"]), 4)
        self.assertEqual(result["metadata"]["warmup_calls_per_backend_and_size"], 1)
        self.assertIn("includes output allocation", result["metadata"]["timing_scope"])
        self.assertIn("JIT warm-up", result["metadata"]["timing_scope"])

    def test_raw_records_metadata_and_fixture_hashes_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_benchmark(root / "first", counts=[4, 9], repeats=2, seed=7, backends=("numpy", "python"))
            second = run_benchmark(root / "second", counts=[4, 9], repeats=1, seed=7, backends=("numpy",))
            with Path(first["paths"]["samples"]).open(newline="", encoding="utf-8") as stream:
                samples = list(csv.DictReader(stream))
            metadata = json.loads(Path(first["paths"]["metadata"]).read_text(encoding="utf-8"))
        self.assertEqual(len(samples), 8)
        self.assertEqual({(int(row["n_particles"]), row["backend"], int(row["repeat"])) for row in samples},
                         {(n, backend, repeat) for n in (4, 9) for backend in ("numpy", "python") for repeat in (1, 2)})
        self.assertEqual(metadata["fixtures"], second["metadata"]["fixtures"])
        self.assertEqual(metadata["seed"], 7)
        self.assertEqual(metadata["dtype"], "float64")
        self.assertEqual(metadata["n_phases"], 12)
        self.assertEqual(metadata["measured_backends"], ["numpy", "python"])
        self.assertEqual(len(metadata["backend_source_sha256"]), 64)
        self.assertEqual(len(metadata["summary"]), 4)
        for row in samples:
            self.assertGreaterEqual(float(row["seconds"]), 0)
            self.assertLess(float(row["max_absolute_error"]), 1e-10)
        for row in second["summary"]:
            self.assertIsNone(row["stdev_seconds"])

    def test_missing_optional_backend_is_explicitly_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(backends, "available_backends", return_value=("python", "numpy")):
                result = run_benchmark(directory, counts=[4], repeats=1, backends=("numpy", "numba"))
        self.assertEqual(result["metadata"]["measured_backends"], ["numpy"])
        self.assertEqual(result["metadata"]["skipped_backends"],
                         [{"backend": "numba", "reason": "optional dependency is not installed"}])
        self.assertEqual({row["backend"] for row in result["samples"]}, {"numpy"})

    def test_incorrect_backend_output_is_rejected_before_results_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with patch.object(backends, "_get_kernel", return_value=lambda v, o: np.zeros((len(o), len(v)))):
                with self.assertRaisesRegex(ArithmeticError, "parity"):
                    run_benchmark(output, counts=[4], repeats=1, backends=("numpy",))
            self.assertFalse(output.exists())

    def test_invalid_configuration_is_rejected_without_output(self):
        cases = ({"counts": []}, {"counts": [0]}, {"counts": [True]}, {"counts": [4, 4]},
                 {"repeats": 0}, {"repeats": True}, {"seed": -1}, {"seed": 1.5},
                 {"backends": []}, {"backends": ["cuda"]}, {"backends": "numpy"},
                 {"backends": ["numpy", "numpy"]})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    run_benchmark(output, **kwargs)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
