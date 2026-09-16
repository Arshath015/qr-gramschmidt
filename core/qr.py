import numpy as np
from .utils import norm

def gram_schmidt_qr(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return the QR decomposition of matrix A using classical Gram‑Schmidt.

    Parameters
    ----------
    A: np.ndarray
        Input matrix of shape (m, n) with m >= n.
    tol: float
        Threshold below which a vector is considered numerically zero.

    Returns
    -------
    Q: np.ndarray
        Orthonormal matrix of shape (m, n).
    R: np.ndarray
        Upper‑triangular matrix of shape (n, n) such that A = Q @ R.
    """
    m, n = A.shape
    if m < n:
        raise ValueError('Number of rows must be >= number of columns for QR decomposition.')
    Q = np.zeros((m, n), dtype=A.dtype)
    R = np.zeros((n, n), dtype=A.dtype)
    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = np.dot(Q[:, j].conj(), A[:, k])
            v -= R[j, k] * Q[:, j]
        R[k, k] = norm(v)
        if R[k, k] < tol:
            raise np.linalg.LinAlgError('Matrix has linearly dependent columns; QR decomposition failed.')
        Q[:, k] = v / R[k, k]
    return Q, R
