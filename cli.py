import argparse
import sys
import numpy as np
from core.qr import gram_schmidt_qr

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='qr-gramschmidt',
        description='Compute QR decomposition of a matrix using classical Gram‑Schmidt.'
    )
    parser.add_argument('matrix_file', help='Path to a .npy file containing a 2‑D numeric array')
    parser.add_argument('-o', '--output', default='qr_output.npz', help='File to store Q and R (NumPy .npz)')
    parser.add_argument('--tol', type=float, default=1e-12, help='Tolerance for detecting linear dependence')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        A = np.load(args.matrix_file)
    except Exception as e:
        sys.stderr.write(f'Failed to load matrix: {e}\n')
        sys.exit(1)
    if A.ndim != 2:
        sys.stderr.write('Input must be a 2‑D array.\n')
        sys.exit(1)
    Q, R = gram_schmidt_qr(A, tol=args.tol)
    np.savez(args.output, Q=Q, R=R)
    print(f'QR decomposition written to {args.output}')

if __name__ == '__main__':
    main()
