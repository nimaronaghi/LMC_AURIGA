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


