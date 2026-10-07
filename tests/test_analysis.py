"""Hand-computed observable expectations and weighted streaming parity."""

import h5py
import numpy as np
import pytest

import lmc_auriga.analysis as analysis


def _snapshot(path, positions, velocities, tags, weights):
    with h5py.File(path, "w") as snapshot:
        snapshot.attrs.update(schema_version=1, position_unit="kpc", velocity_unit="km/s",
                              frame="galactocentric", provenance="synthetic unit-test fixture")
        snapshot["Coordinates"] = np.asarray(positions, dtype=float)
        snapshot["Velocities"] = np.asarray(velocities, dtype=float)
        snapshot["LMC_tag"] = np.asarray(tags, dtype=int)
        snapshot["Weights"] = np.asarray(weights, dtype=float)
    return path


def _table(output, filename):
    return np.genfromtxt(output / filename, delimiter=",", names=True, ndmin=1)


def _two_phase_fixture(tmp_path, monkeypatch):
    # For vx=(0,2,5), the two observer shifts give speeds (1,3,6)
    # and (1,1,4). Large weights at the origin/outside the shell are excluded.
    monkeypatch.setattr(analysis, "observer_velocity", lambda phase, **kwargs:
                        np.array([-1.0 if phase == 0 else 1.0, 0, 0]))
    return _snapshot(tmp_path / "two-phase.h5",
                     [[8, 0, 0], [8, 0, 0], [8, 0, 0], [0, 0, 0], [20, 0, 0]],
                     [[0, 0, 0], [2, 0, 0], [5, 0, 0], [1, 0, 0], [1, 0, 0]],
                     [-1, 1, -1, 1, 1], [1, 2, 3, 100, 100])


def test_phase_averages_additive_weighted_observables_not_particle_mean_speeds(tmp_path, monkeypatch):
    snapshot = _two_phase_fixture(tmp_path, monkeypatch)
    output = tmp_path / "analysis"
    report = analysis.analyze_snapshot(snapshot, output, phase_count=2, chunk_size=2,
                                       speed_max=6, bin_width=1)
    assert report["selected_count"] == 3
    assert report["selected_weight"] == 6
    assert report["configuration"]["phases"] == [0, .5]
    pdf = _table(output, "speed_distribution.csv")
    np.testing.assert_allclose(pdf["total_pdf"], [0, 1/3, 0, 1/6, 1/4, 1/4])
    np.testing.assert_allclose(pdf["mw_pdf"], [0, 1/6, 0, 0, 1/4, 1/4])
    np.testing.assert_allclose(pdf["lmc_pdf"], [0, 1/6, 0, 1/6, 0, 0])
    np.testing.assert_allclose(pdf["mw_pdf"] + pdf["lmc_pdf"], pdf["total_pdf"])
    assert pdf["total_pdf"].sum() == pytest.approx(1)
    # Histogramming each particle's average speed would instead yield this.
    assert not np.allclose(pdf["total_pdf"], [0, 1/6, 1/3, 0, 0, 1/2])
    observables = _table(output, "observables.csv")
    np.testing.assert_allclose(observables["total_tail"], [1, 2/3, 2/3, 1/2, 1/4, 1/4, 0])
    np.testing.assert_allclose(observables["total_eta"], [71/144, 23/144, 23/144, 5/48, 1/24, 1/24, 0])
    for name in ("tail", "eta"):
        np.testing.assert_allclose(observables[f"mw_{name}"] + observables[f"lmc_{name}"],
                                   observables[f"total_{name}"])


def test_histogram_retains_right_edge_and_reports_unplotted_probability(tmp_path, monkeypatch):
    snapshot = _two_phase_fixture(tmp_path, monkeypatch)
    output = tmp_path / "truncated"
    report = analysis.analyze_snapshot(snapshot, output, phase_count=2, speed_max=4, bin_width=1)
    pdf = _table(output, "speed_distribution.csv")
    np.testing.assert_allclose(pdf["total_pdf"], [0, 1/3, 0, 5/12])
    assert report["histogram_overflow_fraction"]["total"] == pytest.approx(1/4)
    assert pdf["total_pdf"].sum() + report["histogram_overflow_fraction"]["total"] == pytest.approx(1)
    observables = _table(output, "observables.csv")
    # Eta and tail remain bin-free, including particles above plotted speed_max.
    assert observables["total_tail"][-1] == pytest.approx(1/4)
    assert observables["total_eta"][-1] == pytest.approx(1/24)


def test_weighted_moments_and_observables_are_independent_of_chunk_partition(tmp_path):
    velocities = np.array([[0, 0, 0], [1, 2, 3], [2, 1, 4], [4, 2, 1], [5, 4, 2.]])
    weights = np.array([.5, 2, 3, 1.5, 4])
    tags = np.array([-1, 1, -1, 1, -1])
    snapshot = _snapshot(tmp_path / "weighted.h5", np.tile([8, 0, 0], (5, 1)),
                         velocities, tags, weights)
    baseline = None
    for chunk_size in (1, 2, 5, 100):
        output = tmp_path / f"chunk-{chunk_size}"
        report = analysis.analyze_snapshot(snapshot, output, chunk_size=chunk_size,
                                           phase_count=3, speed_max=500, bin_width=25)
        # Along +x: (vr,vtheta,vphi)=(vx,-vz,vy), independent of implementation.
        components = velocities[:, [0, 2, 1]] * [1, -1, 1]
        for row, mask in zip(report["anisotropy"], (np.ones(5, dtype=bool), tags == -1, tags == 1)):
            mean = np.average(components[mask], axis=0, weights=weights[mask])
            variance = np.average((components[mask] - mean)**2, axis=0, weights=weights[mask])
            expected_beta = 1 - (variance[1] + variance[2])/(2*variance[0])
            assert row["n"] == mask.sum()
            assert row["weight"] == weights[mask].sum()
            np.testing.assert_allclose(row["mean_spherical_km_s"], mean, atol=1e-14)
            assert row["beta"] == pytest.approx(expected_beta, abs=1e-14)
        numeric = {name: _table(output, name) for name in ("speed_distribution.csv", "observables.csv")}
        if baseline is not None:
            for name, table in numeric.items():
                for column in table.dtype.names:
                    np.testing.assert_allclose(table[column], baseline[name][column], rtol=1e-14, atol=1e-16)
        baseline = numeric


def test_beta_stays_in_host_frame_when_observer_changes(tmp_path):
    velocities = np.vstack((np.eye(3), -np.eye(3))) * [2, 1, 1]
    snapshot = _snapshot(tmp_path / "beta.h5", np.tile([8, 0, 0], (6, 1)),
                         velocities, np.full(6, -1), np.ones(6))
    for vc in (0, 300):
        report = analysis.analyze_snapshot(snapshot, tmp_path / f"vc-{vc}", vc=vc,
                                           phase_count=2, speed_max=500, bin_width=25)
        assert report["anisotropy"][0]["beta"] == pytest.approx(.75)
        assert report["anisotropy"][1]["beta"] == pytest.approx(.75)
        assert report["anisotropy"][2]["beta"] is None
        assert report["anisotropy"][2]["n"] == 0
        assert report["anisotropy"][2]["mean_spherical_km_s"] is None


def test_zero_relative_speed_obeys_strict_eta_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "observer_velocity", lambda phase, **kwargs: np.zeros(3))
    snapshot = _snapshot(tmp_path / "zero.h5", [[8, 0, 0], [8, 0, 0]],
                         [[0, 0, 0], [2, 0, 0]], [-1, 1], [3, 1])
    output = tmp_path / "zero-analysis"
    analysis.analyze_snapshot(snapshot, output, phase_count=1, speed_max=4, bin_width=1)
    observables = _table(output, "observables.csv")
    assert observables["total_tail"][0] == pytest.approx(1/4)
    assert observables["total_eta"][0] == pytest.approx(1/8)
    np.testing.assert_allclose(_table(output, "speed_distribution.csv")["total_pdf"], [3/4, 0, 1/4, 0])
