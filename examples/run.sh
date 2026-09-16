#!/usr/bin/env bash
# Example usage of the QR decomposition CLI
# Generate a random 5x3 matrix and store it as a .npy file
python - <<'PY'
import numpy as np, pathlib
A = np.random.randn(5, 3)
path = pathlib.Path('example_matrix.npy')
np.save(path, A)
print('Saved matrix to', path)
PY

# Run the CLI
./cli.py example_matrix.npy -o result.npz

# Inspect the result
python - <<'PY'
import numpy as np
data = np.load('result.npz')
Q, R = data['Q'], data['R']
print('Q shape:', Q.shape)
print('R shape:', R.shape)
print('Reconstruction error:', np.linalg.norm(Q @ R - np.load('example_matrix.npy')))
PY
