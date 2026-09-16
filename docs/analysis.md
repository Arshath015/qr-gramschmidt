# Numerical Stability of Classical Gram‑Schmidt

The classical Gram‑Schmidt process is straightforward but suffers from loss of orthogonality when the input matrix has columns that are nearly linearly dependent. In double precision, the inner product accumulation error grows proportionally to the condition number of the column set. Our implementation mitigates catastrophic cancellation by:

1. Using a user‑configurable tolerance (`--tol`) to detect degenerate vectors early and abort rather than return a polluted Q.
2. Computing the norm with a single dot product (`v.conj() @ v`) which leverages NumPy's fused‑multiply‑add where available, providing a more accurate L2 norm than `np.linalg.norm` on some architectures.
3. Keeping the algorithm in pure NumPy to benefit from BLAS‑accelerated dot products.

For production‑grade orthogonalization, modified Gram‑Schmidt or Householder reflectors are recommended. This module intentionally showcases the classical variant for educational and lightweight scripting contexts.
