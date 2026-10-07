import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from src import anisotropy


def test_anisotropy_simple_dataset():
    dataset = [
        [0] * 11 + [1.0, 2.0, 3.0],
        [0] * 11 + [4.0, 5.0, 6.0],
        [0] * 11 + [7.0, 8.0, 9.0],
    ]
    assert anisotropy(dataset) == pytest.approx(0.0)


def test_legacy_schema_is_explicit():
    with pytest.raises(ValueError, match="14"):
        anisotropy([[1, 2, 3], [4, 5, 6]])


def test_legacy_uses_centered_components_and_validates_radial_dispersion():
    rows = [[0] * 11 + [10 + i, 20 + 2*i, -30 + 2*i] for i in (-1, 0, 1)]
    assert anisotropy(rows) == pytest.approx(-3)
    for row in rows:
        row[11] = 10
    with pytest.raises(ValueError, match="radial"):
        anisotropy(rows)


