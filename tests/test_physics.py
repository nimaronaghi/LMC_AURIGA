"""Independent geometric identities and closed-form statistical checks."""

import numpy as np
import pytest

from lmc_auriga.physics import (
    anisotropy, mean_inverse_speed, observer_velocity, selection_mask,
    spherical_velocities,
)


def test_spherical_known_polar_vector_and_poles():
    positions = [[1, 0, 1], [0, 0, 3], [0, 0, -3], [2, 0, 0], [0, 2, 0]]
    velocities = [[0, 0, 1], [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4]]
    expected = [[2**-0.5, -(2**-0.5), 0], [4, 2, 3], [-4, -2, 3],
                [2, -4, 3], [3, -4, -2]]
    np.testing.assert_allclose(spherical_velocities(positions, velocities), expected, atol=1e-14)


def test_spherical_preserves_speed_and_rotated_radial_projection():
    rng = np.random.default_rng(734)
    positions, velocities = rng.normal(size=(2, 200, 3))
    components = spherical_velocities(positions, velocities)
    np.testing.assert_allclose(np.sum(components**2, axis=1), np.sum(velocities**2, axis=1), rtol=1e-14)
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rotated = spherical_velocities(positions @ rotation, velocities @ rotation)
    np.testing.assert_allclose(rotated[:, 0], components[:, 0], atol=1e-14)
    np.testing.assert_allclose(np.sum(rotated[:, 1:]**2, axis=1), np.sum(components[:, 1:]**2, axis=1), atol=1e-14)


def test_spherical_coordinate_scale_and_no_input_mutation():
    positions = np.array([[1, 2, 3], [0, 0, -1.0]])
    velocities = np.array([[3, 2, 1], [1, 2, 3.0]])
    before = velocities.copy()
    expected = spherical_velocities(positions, velocities)
    for factor in (1e-300, 1e300):
        np.testing.assert_allclose(spherical_velocities(positions * factor, velocities), expected)
    np.testing.assert_array_equal(velocities, before)
    assert spherical_velocities(np.empty((0, 3)), np.empty((0, 3))).shape == (0, 3)


@pytest.mark.parametrize("positions, velocities", [
    ([[0, 0, 0]], [[1, 2, 3]]), ([[1, 2]], [[1, 2]]),
    ([[1, 0, np.nan]], [[1, 2, 3]]), ([[1, 0, 0]], [[1, np.inf, 3]]),
    ([[1, 0, 0]], [[1, 2, 3], [4, 5, 6]]),
])
def test_spherical_rejects_invalid_geometry(positions, velocities):
    with pytest.raises(ValueError):
        spherical_velocities(positions, velocities)


@pytest.mark.parametrize("radial,tangential,expected", [(1, 1, 0), (2, 1, .75), (1, 2, -3), (1, 0, 1)])
def test_beta_known_covariance_and_streaming_offsets(radial, tangential, expected):
    # Six axial samples have zero cross-covariance and controlled dispersions.
    components = np.vstack((np.eye(3), -np.eye(3))) * [radial, tangential, tangential]
    assert anisotropy(components) == pytest.approx(expected)
    assert anisotropy(components + [100, -200, 300]) == pytest.approx(expected, abs=1e-12)
    assert anisotropy(components * 1e200) == pytest.approx(expected)


@pytest.mark.parametrize("values", [[], [[1, 2, 3]], [[1, 2, 3], [1, 3, 4]],
                                        [[1, 2], [3, 4]], [[0, 0, 0], [np.nan, 1, 2]]])
def test_beta_rejects_undefined_samples(values):
    with pytest.raises(ValueError):
        anisotropy(values)


def test_beta_large_constant_streaming_does_not_erase_radial_dispersion():
    assert anisotropy([[-1, 1e200, 1e200], [1, 1e200, 1e200]]) == 1


def test_selection_inclusive_shell_cone_normalization_and_origin():
    positions = np.array([[6, 0, 0], [10, 0, 0], [8, 8, 0], [8, 0, 8],
                          [-8, 0, 0], [0, 0, 0], [5, 0, 0]], dtype=float)
    np.testing.assert_array_equal(selection_mask(positions, shell=(0, 12), cone_axis=[3, 0, 0]),
                                  [True, True, True, True, False, False, True])
    np.testing.assert_array_equal(selection_mask(positions), [True, True, False, False, True, False, False])
    np.testing.assert_array_equal(selection_mask([[0, 7, 0], [0, -7, 0]], cone_axis=[1, 0, 0], half_angle_deg=90), [True, True])
    assert not selection_mask([[7, 7.01, 0]], shell=(0, 12), cone_axis=[1, 0, 0])[0]


@pytest.mark.parametrize("kwargs", [{"shell": (10, 6)}, {"shell": (-1, 2)},
                                       {"cone_axis": [0, 0, 0]}, {"half_angle_deg": 181},
                                       {"cone_axis": [1, np.inf, 0]}])
def test_selection_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        selection_mask([[7, 0, 0]], **kwargs)


def test_observer_known_phase_circular_speed_and_periodicity():
    expected = np.array([11.1, 242.24, 7.25]) + 29.8 * np.array([.9931, .1170, -.01032])
    np.testing.assert_allclose(observer_velocity(.218, vc=230), expected, atol=1e-13)
    np.testing.assert_allclose(observer_velocity(.218 + 4, vc=230), expected, atol=1e-12)
    np.testing.assert_allclose(observer_velocity(.468) - [11.1, 232.24, 7.25],
                               29.8 * np.array([-.067, .4927, -.8676]), atol=1e-13)
    np.testing.assert_allclose(observer_velocity(.1, vc=250) - observer_velocity(.1, vc=220), [0, 30, 0], atol=1e-13)


def test_observer_basis_rows_and_galilean_relative_velocity():
    # Rows are local axes in simulation coordinates: local +U points sim +y.
    basis = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    local = observer_velocity(.218)
    observer = observer_velocity(.218, basis=basis)
    np.testing.assert_allclose(observer, [-local[1], local[0], local[2]])
    np.testing.assert_allclose(np.linalg.norm(observer), np.linalg.norm(local))
    particle = np.array([100, 250, 20.0])
    boost = np.array([-40, 120, 70.0])
    np.testing.assert_allclose((particle + boost) - (observer + boost), particle - observer)
    assert np.linalg.norm(observer - observer) == 0


@pytest.mark.parametrize("kwargs", [{"phase": np.nan}, {"phase": [0]}, {"phase": 0, "vc": -1},
                                       {"phase": 0, "basis": np.diag([1, 1, -1])},
                                       {"phase": 0, "basis": np.eye(3)*2}])
def test_observer_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        observer_velocity(**kwargs)


def test_inverse_speed_exact_weighted_atoms_and_strict_threshold():
    np.testing.assert_allclose(mean_inverse_speed([0, 2, 4], [0, 2, 3, 4, 10], weights=[4, 2, 2]),
                               [3/16, 1/16, 1/16, 0, 0])
    np.testing.assert_allclose(mean_inverse_speed([3, 3, 3], [0, 2.9, 3]), [1/3, 1/3, 0])
    np.testing.assert_array_equal(mean_inverse_speed([0, 0], 0), [0])
    np.testing.assert_allclose(mean_inverse_speed([2, 4], 0, weights=[1e308, 1e308]), [3/8])
    # Threshold ordering is retained and unsampled tail probability stays zero.
    np.testing.assert_allclose(mean_inverse_speed([4, 2], [4, 0, 2]), [0, 3/8, 1/8])


def test_inverse_speed_maxwell_distribution_quadrature():
    # Independent continuous reference for isotropic 3D Gaussian components:
    # eta(vmin)=sqrt(2/pi)/sigma * exp(-vmin^2/(2*sigma^2)).
    sigma = 150.0
    speed = (np.arange(5000) + .5) * .002 * sigma
    probability = speed**2 * np.exp(-.5*(speed/sigma)**2)
    thresholds = sigma * np.arange(4)
    expected = np.sqrt(2/np.pi)/sigma * np.exp(-.5*(thresholds/sigma)**2)
    np.testing.assert_allclose(mean_inverse_speed(speed, thresholds, weights=probability), expected, rtol=3e-6)


def test_annual_inverse_speed_is_average_of_statistics_not_mean_speed():
    instantaneous = [mean_inverse_speed([speed], [1]) for speed in (0.5, 2)]
    assert np.mean(instantaneous) == pytest.approx(.25)
    assert mean_inverse_speed([1.25], 1)[0] == pytest.approx(.8)


@pytest.mark.parametrize("speeds,vmin,weights", [([], 0, None), ([-1], 0, None), ([1], -1, None),
    ([1], 0, [0]), ([1], 0, [-1]), ([1], 0, [1, 2]), ([np.inf], 0, None),
    ([1], [[0]], None), ([1], np.nan, None)])
def test_inverse_speed_invalid_inputs(speeds, vmin, weights):
    with pytest.raises(ValueError):
        mean_inverse_speed(speeds, vmin, weights)


@pytest.mark.parametrize("function,args,kwargs", [
    (spherical_velocities, ([[1+1j, 0, 0]], [[1, 2, 3]]), {}),
    (anisotropy, ([[1+1j, 2, 3], [2, 3, 4]],), {}),
    (selection_mask, ([[1, 0, 0]],), {"cone_axis": [1+1j, 0, 0]}),
    (observer_velocity, (1+1j,), {}),
    (observer_velocity, (0,), {"basis": np.eye(3, dtype=complex)}),
    (mean_inverse_speed, ([1+1j], 0), {}),
    (mean_inverse_speed, ([1], 0), {"weights": [1+1j]}),
])
def test_complex_physical_inputs_are_rejected_not_silently_projected(function, args, kwargs):
    with pytest.raises(ValueError, match="real numeric"):
        function(*args, **kwargs)
