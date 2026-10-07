"""Validated halo kinematics and a simple annual observer-velocity model.

Positions must already be centered on the intended Galactic origin. Velocities
must already be in the corresponding rest frame. This module does not infer
simulation units, subtract a halo bulk velocity, or convert raw Auriga data.
"""

import numpy as np


def _real_array(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(array, dtype=np.float64)


def _vectors(values, name):
    """Return finite float64 vectors without modifying caller-owned arrays."""
    array = _real_array(values, name)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unit_vectors(vectors):
    """Normalize nonzero rows without squaring large or tiny coordinates."""
    scale = np.max(np.abs(vectors), axis=1)
    if np.any(scale == 0):
        raise ValueError("positions at the origin have no spherical direction")
    scaled = vectors / scale[:, None]
    return scaled / np.hypot.reduce(scaled, axis=1)[:, None]


def spherical_velocities(positions, velocities):
    """Return ``(v_r, v_theta, v_phi)`` for matching ``(N, 3)`` arrays.

    Theta is the polar angle measured from +z, and phi is azimuth from +x
    toward +y. At either z-axis pole, where azimuth is undefined, choose phi=0:
    e_phi=+y and e_theta=+x at the north pole or -x at the south pole. The
    resulting basis is orthonormal and preserves speed. Individual tangential
    component statistics at poles depend on this declared basis convention.
    The origin is rejected. Output velocity units equal input velocity units.
    """
    positions = _vectors(positions, "positions")
    velocities = _vectors(velocities, "velocities")
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have matching shapes")
    radial = _unit_vectors(positions)
    # Preserve tiny transverse directions even when |z| is very large.
    transverse_scale = np.max(np.abs(positions[:, :2]), axis=1)
    transverse = np.divide(
        positions[:, :2], transverse_scale[:, None],
        out=np.zeros_like(positions[:, :2]), where=transverse_scale[:, None] > 0,
    )
    rho = np.hypot(transverse[:, 0], transverse[:, 1])
    phi = np.zeros_like(positions)
    nonpole = rho > 0
    phi[nonpole, 0] = -transverse[nonpole, 1] / rho[nonpole]
    phi[nonpole, 1] = transverse[nonpole, 0] / rho[nonpole]
    phi[~nonpole, 1] = 1.0
    theta = np.cross(phi, radial)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.column_stack([
            np.einsum("ij,ij->i", velocities, axis)
            for axis in (radial, theta, phi)
        ])
    if not np.all(np.isfinite(result)):
        raise ValueError("spherical velocity projection exceeds float64 range")
    return result


def anisotropy(components):
    """Estimate beta from centered ``(v_r, v_theta, v_phi)`` samples.

    ``beta = 1 - (s_theta**2 + s_phi**2) / (2*s_r**2)``, with sample
    variances (ddof=1). At least two particles and positive radial dispersion
    are required. All components must use the same velocity units and frame.
    Compute halo anisotropy in the Galactocentric frame, before observer boosts.

    This ratio estimator is not unbiased at finite N. For independent Gaussian
    components and N>3, E[beta_hat] = 1-(1-beta)*(N-1)/(N-3). No universal
    bias correction is applied. Beta=0 alone does not prove isotropy.
    """
    components = _vectors(components, "components")
    if len(components) < 2:
        raise ValueError("anisotropy requires at least two particles")
    # Remove constant streaming offsets before choosing the scale. Otherwise
    # a huge constant tangential mean can underflow a finite radial variance.
    with np.errstate(over="ignore"):
        shifted = components - components[0]
    if not np.all(np.isfinite(shifted)):
        scaled = components / np.max(np.abs(components))
        shifted = scaled - scaled[0]
    scale = np.max(np.abs(shifted))
    if scale == 0:
        raise ValueError("radial velocity dispersion must be positive")
    variances = np.var(shifted / scale, axis=0, ddof=1)
    if variances[0] <= 0:
        raise ValueError("radial velocity dispersion must be positive")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        beta = 1.0 - 0.5 * (variances[1] / variances[0] + variances[2] / variances[0])
    if not np.isfinite(beta):
        raise ValueError("anisotropy exceeds float64 range")
    return float(beta)


def selection_mask(positions, shell=(6.0, 10.0), cone_axis=None, half_angle_deg=45.0):
    """Select an inclusive radial shell and optional inclusive angular cone.

    Shell radii use the same units as positions (normally physical kpc).
    ``cone_axis`` points toward the selected region and need not be normalized.
    Origins are always excluded. Cone boundaries allow eight float64 epsilons
    in cosine to retain mathematically inclusive cuts after roundoff.
    """
    positions = _vectors(positions, "positions")
    shell = _real_array(shell, "shell")
    if (shell.shape != (2,) or not np.all(np.isfinite(shell))
            or shell[0] < 0 or shell[1] < shell[0]):
        raise ValueError("shell must be two finite radii with 0 <= inner <= outer")
    angle = _real_array(half_angle_deg, "half_angle_deg")
    if angle.ndim != 0 or not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("half_angle_deg must be finite and between 0 and 180")
    with np.errstate(over="ignore"):
        radii = np.hypot.reduce(positions, axis=1)
    selected = (radii > 0) & (radii >= shell[0]) & (radii <= shell[1])
    if cone_axis is None:
        return selected
    axis = _real_array(cone_axis, "cone_axis")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or not np.any(axis):
        raise ValueError("cone_axis must be a finite nonzero 3-vector")
    axis = _unit_vectors(axis[None, :])[0]
    candidates = np.flatnonzero(selected)
    cosine = _unit_vectors(positions[candidates]) @ axis
    threshold = np.cos(np.deg2rad(float(angle)))
    selected[candidates] &= cosine >= threshold - 8 * np.finfo(np.float64).eps
    return selected


def observer_velocity(phase, vc=220.0, basis=None):
    """Return an approximate observer velocity in simulation coordinates, km/s.

    ``phase`` is fractional year measured from January 1; phase=0.218 sets
    solar ecliptic longitude to zero in this circular-orbit approximation.
    The local Galactic axes are U toward the Galactic center, V along Galactic
    rotation, and W toward the north Galactic pole. The model adds circular
    speed ``vc``, solar peculiar velocity (11.1, 12.24, 7.25) km/s, and the
    original notebook's rounded 29.8 km/s annual orbital approximation. It is
    periodic, not a precision ephemeris, and omits Earth's daily rotation.

    Each row of ``basis`` is one local Galactic unit axis expressed in the
    simulation coordinate system. It must be a proper orthogonal 3x3 matrix;
    identity is the default. For row vectors, v_sim = v_local @ basis.
    Observer-frame particle velocities are v_particle - observer_velocity(...).

    The orbital signs, velocity basis and phase follow Savage, Freese & Gondolo
    (2006), Eqs. 13--15: https://arxiv.org/abs/astro-ph/0607121 . Their velocity
    basis must not be confused with the position basis used by some papers.
    """
    phase = _real_array(phase, "phase")
    vc = _real_array(vc, "vc")
    if phase.ndim != 0 or not np.isfinite(phase):
        raise ValueError("phase must be a finite scalar")
    if vc.ndim != 0 or not np.isfinite(vc) or vc < 0:
        raise ValueError("vc must be a finite nonnegative circular speed")
    if basis is None:
        basis = np.eye(3)
    basis = _real_array(basis, "basis")
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("basis must be a finite 3x3 matrix")
    if (not np.allclose(basis @ basis.T, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(basis), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("basis must be orthonormal and right-handed")
    longitude = 2 * np.pi * (float(phase) % 1.0 - 0.218)
    sine, cosine = np.sin(longitude), np.cos(longitude)
    local = np.array([11.1, float(vc) + 12.24, 7.25]) + 29.8 * np.array([
        -0.067 * sine + 0.9931 * cosine,
        0.4927 * sine + 0.1170 * cosine,
        -0.8676 * sine - 0.01032 * cosine,
    ])
    with np.errstate(over="ignore", invalid="ignore"):
        result = local @ basis
    if not np.all(np.isfinite(result)):
        raise ValueError("observer velocity exceeds float64 range")
    return result


def mean_inverse_speed(speeds, vmin, weights=None):
    """Return the empirical mean inverse speed at each threshold in ``vmin``.

    eta(vmin) = sum_i w_i * I(speed_i > vmin) / speed_i / sum_i w_i.
    The denominator includes all supplied particles, not only those above
    threshold. Weights must be finite, nonnegative, and have positive total;
    equal weights are the default. Speeds and thresholds must be nonnegative.
    Output is a one-dimensional array, including for a scalar threshold, with
    inverse input-speed units (s/km when speeds are in km/s).

    The strict threshold convention excludes exactly zero speeds at vmin=0;
    their weights still enter the denominator. Thus no 1/0 or arbitrary speed
    floor is introduced. A discrete atom at zero is not assigned the divergent
    zero-threshold integral that a non-strict convention could imply.
    """
    speeds = _real_array(speeds, "speeds")
    thresholds = np.atleast_1d(_real_array(vmin, "vmin"))
    if (speeds.ndim != 1 or len(speeds) == 0
            or not np.all(np.isfinite(speeds)) or np.any(speeds < 0)):
        raise ValueError("speeds must be a nonempty finite nonnegative 1D array")
    if (thresholds.ndim != 1 or not np.all(np.isfinite(thresholds))
            or np.any(thresholds < 0)):
        raise ValueError("vmin must be finite nonnegative scalars or a 1D array")
    if weights is None:
        weights = np.ones_like(speeds)
    weights = _real_array(weights, "weights")
    if (weights.shape != speeds.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("weights must match speeds, be finite/nonnegative, and have positive total")
    weights = weights / weights.max()
    weights = weights / weights.sum()
    if len(thresholds) == 0:
        return np.empty(0, dtype=np.float64)
    # Only speeds above the smallest requested threshold can contribute.
    usable = (speeds > thresholds.min()) & (weights > 0)
    order = np.argsort(speeds[usable])
    ordered_speeds = speeds[usable][order]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        terms = weights[usable][order] / ordered_speeds
        tails = np.concatenate((np.cumsum(terms[::-1])[::-1], [0.0]))
    result = tails[np.searchsorted(ordered_speeds, thresholds, side="right")]
    if not np.all(np.isfinite(result)):
        raise ValueError("mean inverse speed exceeds float64 range")
    return result
