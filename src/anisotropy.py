"""Compatibility adapter for the historical 14-column anisotropy API.

New code should call ``lmc_auriga.physics.anisotropy`` with an explicit N x 3
array of spherical velocity components. No notebook DataFrame schema is guessed.
"""

import numpy as np

from .lmc_auriga.physics import anisotropy as _anisotropy


def anisotropy(dataset):
    """Compute beta from historical zero-based columns 11, 12, and 13.

    Require at least fourteen columns, preserving the original public schema.
    Component validation and the estimator are shared with the scientific core.
    """
    dataset = np.asarray(dataset)
    if dataset.ndim != 2 or dataset.shape[1] < 14:
        raise ValueError("legacy dataset must have shape (N, >=14); velocity columns are 11:14")
    return _anisotropy(dataset[:, 11:14])
