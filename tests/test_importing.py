"""Legacy import fidelity, bounded validation, and atomic no-clobber publication."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lmc_auriga import importing
from lmc_auriga.importing import import_legacy_snapshot
from lmc_auriga.io import iter_snapshot, read_metadata


class LegacyImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source, self.destination = self.root / "legacy.hdf5", self.root / "processed.hdf5"

    def fixture(self, count=7, weights=True):
        with h5py.File(self.source, "w") as source:
            source["Coordinates"] = np.arange(count * 3, dtype=np.float32).reshape(count, 3) / 4
            source["Velocities"] = np.arange(count * 3, dtype=np.float64).reshape(count, 3) - 15
            source["LMC_tag"] = np.resize(np.array([-1, 4, 9, -1, .5, 2, -1], dtype=np.float64), count)
            source["Coordinates"].attrs["legacy_note"] = "already physical"
            source["LMC_tag"].attrs["legacy_note"] = "positive identifiers distinguish tagged material"
            if weights:
                source["Weights"] = np.arange(1, count + 1, dtype=np.float32)
            source.attrs["halo_id"] = 49
            source.attrs["legacy_array"] = np.array([1, 2, 3], dtype=np.int32)
            source.attrs["comment"] = "source metadata retained"

    def import_snapshot(self, **kwargs):
        return import_legacy_snapshot(self.source, self.destination, provenance="user-confirmed processed sample",
                                      confirmed_processed_frame=True, chunk_size=3, **kwargs)

    def assert_no_partial_output(self):
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(f".{self.destination.name}.*.tmp")), [])

    def test_import_preserves_fields_weights_tags_metadata_and_source(self):
        self.fixture()
        original_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        metadata = self.import_snapshot()
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), original_hash)
        self.assertEqual(metadata["source_file_sha256"], original_hash)
        self.assertTrue(metadata["confirmed_processed_frame"])
        self.assertEqual(metadata["position_unit"], "kpc")
        self.assertEqual(metadata["velocity_unit"], "km/s")
        self.assertEqual(metadata["frame"], "galactocentric")
        self.assertEqual((metadata["mw_count"], metadata["lmc_count"]), (3, 4))
        self.assertIn("copied unchanged", metadata["weight_policy"])
        original_attributes = json.loads(metadata["original_attributes_json"])
        self.assertEqual(original_attributes, {"halo_id": 49, "legacy_array": [1, 2, 3],
                                              "comment": "source metadata retained"})
        with h5py.File(self.source, "r") as source, h5py.File(self.destination, "r") as destination:
            for name in ("Coordinates", "Velocities", "Weights"):
                np.testing.assert_array_equal(destination[name][...], source[name][...])
                self.assertEqual(destination[name].dtype, source[name].dtype)
            np.testing.assert_array_equal(destination["Original_LMC_tag"][...], source["LMC_tag"][...])
            self.assertEqual(destination["Original_LMC_tag"].dtype, source["LMC_tag"].dtype)
            np.testing.assert_array_equal(destination["LMC_tag"][...], [-1, 1, 1, -1, 1, 1, -1])
            self.assertEqual(destination["LMC_tag"].dtype, np.dtype("int8"))
            self.assertEqual(destination["Coordinates"].attrs["legacy_note"], "already physical")
            self.assertIn("positive identifiers", destination["Original_LMC_tag"].attrs["legacy_note"])
        self.assertEqual(read_metadata(self.destination), metadata)

    def test_missing_weights_remain_implicit_number_weights_and_empty_import_is_valid(self):
        self.fixture(weights=False)
        metadata = self.import_snapshot()
        self.assertFalse(metadata["has_weights"])
        self.assertIn("unit number weights", metadata["weight_policy"])
        for _, _, _, weights in iter_snapshot(self.destination, chunk_size=2):
            np.testing.assert_array_equal(weights, np.ones(len(weights)))
        self.fixture(count=0, weights=False)
        empty = self.root / "empty.hdf5"
        metadata = import_legacy_snapshot(self.source, empty, provenance="empty test", confirmed_processed_frame=True)
        self.assertEqual(metadata["n_particles"], 0)
        self.assertEqual(list(iter_snapshot(empty)), [])

    def test_reads_particle_fields_in_bounded_slices(self):
        self.fixture(count=8)
        original_getitem = h5py.Dataset.__getitem__
        reads = []

        def recording_read(dataset, key):
            if os.path.normcase(os.path.abspath(dataset.file.filename)) == os.path.normcase(os.path.abspath(self.source)):
                self.assertIsInstance(key, slice)
                self.assertLessEqual(key.stop - key.start, 3)
                reads.append((dataset.name, key.start, key.stop))
            return original_getitem(dataset, key)

        with patch.object(h5py.Dataset, "__getitem__", new=recording_read):
            self.import_snapshot()
        self.assertEqual(len(reads), 4 * 3)
        self.assertEqual({(start, stop) for _, start, stop in reads}, {(0, 3), (3, 6), (6, 8)})

    def test_late_invalid_values_never_publish_or_leave_partial_files(self):
        for field, value in (("Coordinates", np.nan), ("Velocities", np.inf),
                             ("LMC_tag", 0), ("LMC_tag", -2), ("LMC_tag", np.nan),
                             ("Weights", 0), ("Weights", -1), ("Weights", np.nan)):
            with self.subTest(field=field, value=value):
                self.fixture()
                with h5py.File(self.source, "r+") as source:
                    source[field][6] = value
                with patch.object(importing, "_publish_no_replace", side_effect=AssertionError("published invalid data")):
                    with self.assertRaisesRegex(ValueError, field):
                        self.import_snapshot()
                self.assert_no_partial_output()

    def test_explicit_confirmation_and_valid_metadata_are_required(self):
        self.fixture()
        for confirmation in (False, None, 1, "true", np.bool_(True)):
            with self.subTest(confirmation=confirmation), self.assertRaisesRegex(ValueError, "explicit"):
                import_legacy_snapshot(self.source, self.destination, provenance="test",
                                       confirmed_processed_frame=confirmation)
        for provenance in ("", "  ", None):
            with self.subTest(provenance=provenance), self.assertRaisesRegex(ValueError, "provenance"):
                import_legacy_snapshot(self.source, self.destination, provenance=provenance,
                                       confirmed_processed_frame=True)
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaisesRegex(ValueError, "chunk_size"):
                import_legacy_snapshot(self.source, self.destination, provenance="test",
                                       confirmed_processed_frame=True, chunk_size=size)
        self.assert_no_partial_output()

    def test_same_or_existing_destination_cannot_be_overwritten(self):
        self.fixture()
        original_hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "different files"):
            import_legacy_snapshot(self.source, self.source, provenance="test", confirmed_processed_frame=True)
        self.destination.write_bytes(b"existing file must survive")
        with self.assertRaises(FileExistsError):
            self.import_snapshot()
        self.assertEqual(self.destination.read_bytes(), b"existing file must survive")
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), original_hash)

    def test_racing_destination_creation_is_protected(self):
        self.fixture()
        publish = importing._publish_no_replace

        def concurrent_publication(temporary, destination):
            with destination.open("xb") as stream:
                stream.write(b"concurrently created file")
            publish(temporary, destination)

        with patch.object(importing, "_publish_no_replace", side_effect=concurrent_publication):
            with self.assertRaises(FileExistsError):
                self.import_snapshot()
        self.assertEqual(self.destination.read_bytes(), b"concurrently created file")
        self.assertEqual(list(self.root.glob(f".{self.destination.name}.*.tmp")), [])

    def test_shape_and_optional_weight_validation(self):
        for field, data in (("Coordinates", np.ones((7, 2))), ("Velocities", np.ones((6, 3))),
                            ("LMC_tag", np.ones((7, 1))), ("Weights", np.ones(6)),
                            ("Velocities", np.ones((7, 3), dtype=complex))):
            with self.subTest(field=field, shape=data.shape):
                self.fixture()
                with h5py.File(self.source, "r+") as source:
                    del source[field]
                    source[field] = data
                with self.assertRaisesRegex(ValueError, field):
                    self.import_snapshot()
                self.assert_no_partial_output()


if __name__ == "__main__":
    unittest.main()
