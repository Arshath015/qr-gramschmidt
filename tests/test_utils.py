import numpy as np
from core.utils import norm

def test_norm_simple():
    v = np.array([3., 4.])
    assert np.isclose(norm(v), 5.0)

def test_norm_zero():
    v = np.zeros(5)
    assert norm(v) == 0.0
