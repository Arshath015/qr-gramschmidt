import numpy as np
import pytest
from core.qr import gram_schmidt_qr

def test_basic_qr():
    A = np.array([[1., 1.], [1., -1.], [1., 0.]])
    Q, R = gram_schmidt_qr(A)
    # Q should be orthonormal
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-10)
    # Reconstruct A
    assert np.allclose(Q @ R, A, atol=1e-10)

def test_dependent_columns():
    A = np.array([[1., 2.], [2., 4.], [3., 6.]])  # second column = 2 * first
    with pytest.raises(np.linalg.LinAlgError):
        gram_schmidt_qr(A)

def test_tolerance_edge_case():
    A = np.array([[1e-13, 0.], [0., 1.]])
    # First column is below default tolerance, should raise
    with pytest.raises(np.linalg.LinAlgError):
        gram_schmidt_qr(A)
