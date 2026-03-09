import numpy as np
import scipy.sparse as sp


def assemble_poisson_2d(N):
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csr')
    I = sp.eye(N, format='csr')
    A = sp.kron(I, T, format='csr') + sp.kron(T, I, format='csr')
    return A


def assemble_rhs(f, N):
    h = 1.0 / (N + 1)
    return (h ** 2) * f.ravel()


def get_grid_points(N):
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1.0 - h, N)
    y = np.linspace(h, 1.0 - h, N)
    return np.meshgrid(x, y)


def validate_matrix(A, N):
    n = N * N
    assert A.shape == (n, n)
    diff = A - A.T
    assert diff.nnz == 0 or abs(diff).max() < 1e-14
    assert np.allclose(A.diagonal(), 4.0)
    assert np.diff(A.indptr).max() <= 5
    return True
