"""Utility functions related to velocity anisotropy."""

from typing import Sequence


def _sample_std(values: Sequence[float]) -> float:
    """Return the sample standard deviation of *values*."""
    n = len(values)
    if n < 2:
        raise ValueError("sample_std requires at least two data points")
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return var ** 0.5


def anisotropy(dataset: Sequence[Sequence[float]]) -> float:
    """Compute the anisotropy parameter β for *dataset*.

    Parameters
    ----------
    dataset : sequence of sequences
        Each inner sequence must have at least 14 elements. Columns 11, 12 and
        13 correspond to ``v_r``, ``v_theta`` and ``v_phi`` respectively
        (0-based indexing).

    Returns
    -------
    float
        The velocity anisotropy parameter β.
    """

    vr = [row[11] for row in dataset]
    vth = [row[12] for row in dataset]
    vphi = [row[13] for row in dataset]

    sigma_vr = _sample_std(vr)
    sigma_vth = _sample_std(vth)
    sigma_vphi = _sample_std(vphi)

    return 1 - (sigma_vth ** 2 + sigma_vphi ** 2) / (2 * sigma_vr ** 2)

