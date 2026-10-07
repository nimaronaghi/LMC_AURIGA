"""Equivalent kernels for speeds relative to an explicit observer velocity.

Inputs already share one Cartesian frame and velocity unit. The output retains
every observer phase; averaging a particle's speeds is not equivalent to
averaging the phase-resolved speed distributions.
"""

import importlib.util
import math

import numpy as np


BACKENDS = ("python", "numpy", "numba")
_COMPILED_NUMBA_KERNEL = None


def available_backends():
    """List implemented backends whose dependencies are installed.

    Numba remains an optional import. A broken Numba installation raises its
    actual import/compilation error when selected instead of silently falling
    back to a different backend.
    """
    available = ["python", "numpy"]
    if importlib.util.find_spec("numba") is not None:
        available.append("numba")
    return tuple(available)


def _velocity_array(values, name):
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric velocities")
    result = np.ascontiguousarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite velocities")
    return result


def _prepare_inputs(velocities, observer_velocities):
    velocities = _velocity_array(velocities, "velocities")
    observers = _velocity_array(observer_velocities, "observer_velocities")
    if observers.shape[0] == 0:
        raise ValueError("observer_velocities must contain at least one phase")
    return velocities, observers


def _python_kernel(velocities, observers):
    """Scalar reference implementation, intended for correctness comparisons."""
    output = np.empty((len(observers), len(velocities)), dtype=np.float64)
    for phase, observer in enumerate(observers):
        for index, velocity in enumerate(velocities):
            dx, dy, dz = (float(v) - float(o) for v, o in zip(velocity, observer))
            output[phase, index] = math.hypot(math.hypot(dx, dy), dz)
    return output


def _numpy_kernel(velocities, observers):
    # Retain only one phase's three-component working array, rather than a
    # (T, N, 3) temporary. The requested (T, N) output is allocated here.
    output = np.empty((len(observers), len(velocities)), dtype=np.float64)
    for phase, observer in enumerate(observers):
        delta = velocities - observer
        output[phase] = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
    return output


def _numba_kernel():
    global _COMPILED_NUMBA_KERNEL
    if _COMPILED_NUMBA_KERNEL is None:
        try:
            from numba import njit, prange
        except ModuleNotFoundError as error:
            if error.name != "numba":
                raise
            raise ImportError(
                "Numba is not installed; install the parallel extra with "
                "python -m pip install '.[parallel]' from the project directory."
            ) from error

        @njit(parallel=True, cache=True)
        def kernel(velocities, observers):
            output = np.empty((observers.shape[0], velocities.shape[0]), dtype=np.float64)
            for index in prange(velocities.shape[0]):
                for phase in range(observers.shape[0]):
                    dx = velocities[index, 0] - observers[phase, 0]
                    dy = velocities[index, 1] - observers[phase, 1]
                    dz = velocities[index, 2] - observers[phase, 2]
                    # Each parallel iteration owns distinct [phase, index]
                    # entries; no shared counters or output compaction races.
                    output[phase, index] = math.hypot(math.hypot(dx, dy), dz)
            return output

        _COMPILED_NUMBA_KERNEL = kernel
    return _COMPILED_NUMBA_KERNEL


def _get_kernel(backend):
    if backend == "python":
        return _python_kernel
    if backend == "numpy":
        return _numpy_kernel
    if backend == "numba":
        return _numba_kernel()
    raise ValueError(f"backend must be one of {BACKENDS}")


def shifted_speeds(velocities, observer_velocities, backend="numpy"):
    """Return float64 speeds of shape ``(T, N)`` in the input velocity unit.

    ``velocities`` has shape ``(N, 3)`` and ``observer_velocities`` has shape
    ``(T, 3)``. Both must be finite real arrays in the same frame. Empty particle
    sets are supported; at least one observer phase is required. Inputs are
    never mutated. ``numba`` is optional; ``python`` is a scalar reference.
    """
    velocities, observers = _prepare_inputs(velocities, observer_velocities)
    result = _get_kernel(backend)(velocities, observers)
    if not np.isfinite(result).all():
        raise ValueError("relative speeds exceed the finite float64 range")
    return result
