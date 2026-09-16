# QR‑GramSchmidt
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: >=3.9](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)

Fast, dependency‑light QR decomposition using the classical Gram‑Schmidt algorithm.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Analysis Document](#analysis-document)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
`qr-gramschmidt` provides a tiny CLI that reads a NumPy ``.npy`` matrix, computes its QR decomposition via classical Gram‑Schmidt, and writes ``Q`` and ``R`` to a ``.npz`` archive.

## Tech Stack
- Python >=3.9
- NumPy (numerical kernels)
- pytest (test runner)

## Architecture
```text
+----------+   args   +-------+   load   +--------+   QR   +----------+
| cli.py   |--------->| core/ |--------->| numpy  |------>| result   |
+----------+          | qr.py |          +--------+       +----------+
```
The CLI parses arguments, loads the matrix, calls `core.qr.gram_schmidt_qr`, and persists the result.

## Theoretical Background
The classical Gram‑Schmidt process orthogonalizes a set of vectors \{a₁,…,a_n\} by iteratively subtracting projections onto previously computed orthonormal basis vectors. For each column k:
```
v_k = a_k - Σ_{j<k} (q_jᵀ a_k) q_j
r_{kk} = ||v_k||
q_k = v_k / r_{kk}
```
The resulting matrices satisfy A = Q R with Qᵀ Q = I. While simple, the method is numerically unstable for ill‑conditioned column sets because rounding errors accumulate in the subtraction step, degrading orthogonality. Users can control the tolerance for detecting near‑zero norms via the ``--tol`` flag.

## Installation
```bash
git clone https://github.com/yourorg/qr-gramschmidt.git
cd qr-gramschmidt
pip install -r requirements.txt
```
*No external compiled extensions are required.*

## Usage
```bash
# Create a matrix file
python - <<PY
import numpy as np
np.save('my_matrix.npy', np.random.rand(6,4))
PY

# Compute QR
./cli.py my_matrix.npy -o decomposition.npz

# Verify
python - <<PY
import numpy as np
mat = np.load('my_matrix.npy')
res = np.load('decomposition.npz')
print('Error:', np.linalg.norm(res['Q'] @ res['R'] - mat))
PY
```

## API Reference
- `core.qr.gram_schmidt_qr(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]`
  Computes QR via classical Gram‑Schmidt.
- `core.utils.norm(v: np.ndarray) -> float`
  Euclidean norm helper used by the algorithm.

## Analysis Document
See the detailed numerical‑stability discussion in [`docs/analysis.md`](docs/analysis.md).

## Testing
```bash
pytest -q
```
The test suite covers basic correctness, detection of linearly dependent columns, and edge‑case tolerance handling.

## Limitations
- Not suitable for highly ill‑conditioned matrices; orthogonality loss may be significant.
- Only works for real or complex dense arrays; sparse support is absent.
- Requires m ≥ n (more rows than columns).

## Roadmap
- Add optional `--modified` flag to switch to Modified Gram‑Schmidt.
- Provide a `--householder` mode for robust decomposition.
- Emit optional CSV/JSON summaries of singular values.

## License
MIT License
