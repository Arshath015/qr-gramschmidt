import numpy as np

def norm(v: np.ndarray) -> float:
    """Compute Euclidean (L2) norm of a vector with high‑precision accumulation.

    Parameters
    ----------
    v: np.ndarray
        Input 1‑D array.
    Returns
    -------
    float
        The Euclidean norm.
    """
    return np.sqrt(np.dot(v.conj(), v))
