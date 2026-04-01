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


def assemble_variable_poisson_2d(N: int, D: np.ndarray) -> sp.csr_matrix:
    h = 1.0 / (N + 1)
    n = N * N

    D_padded = np.zeros((N + 2, N + 2))
    D_padded[1:-1, 1:-1] = D.reshape(N, N)
    D_padded[0, :] = D_padded[1, :]
    D_padded[-1, :] = D_padded[-2, :]
    D_padded[:, 0] = D_padded[:, 1]
    D_padded[:, -1] = D_padded[:, -2]

    rows, cols, vals = [], [], []

    for j in range(N):
        for i in range(N):
            idx = j * N + i
            ip, jp = i + 1, j + 1

            d_e = 0.5 * (D_padded[jp, ip] + D_padded[jp, ip + 1])
            d_w = 0.5 * (D_padded[jp, ip] + D_padded[jp, ip - 1])
            d_n = 0.5 * (D_padded[jp, ip] + D_padded[jp + 1, ip])
            d_s = 0.5 * (D_padded[jp, ip] + D_padded[jp - 1, ip])

            diag = (d_e + d_w + d_n + d_s) / (h * h)
            rows.append(idx)
            cols.append(idx)
            vals.append(diag)

            if i > 0:
                rows.append(idx)
                cols.append(idx - 1)
                vals.append(-d_w / (h * h))
            if i < N - 1:
                rows.append(idx)
                cols.append(idx + 1)
                vals.append(-d_e / (h * h))
            if j > 0:
                rows.append(idx)
                cols.append(idx - N)
                vals.append(-d_s / (h * h))
            if j < N - 1:
                rows.append(idx)
                cols.append(idx + N)
                vals.append(-d_n / (h * h))

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def generate_diffusion_coefficient(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                                   pattern: str = 'smooth') -> np.ndarray:
    if pattern == 'smooth':
        cx, cy = rng.uniform(0.2, 0.8, 2)
        sigma = rng.uniform(0.15, 0.4)
        base = 1.0 + 4.0 * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
        return base

    elif pattern == 'discontinuous':
        D = np.ones_like(X)
        cx, cy = rng.uniform(0.3, 0.7, 2)
        r = rng.uniform(0.1, 0.3)
        mask = (X - cx)**2 + (Y - cy)**2 < r**2
        D[mask] = rng.uniform(10.0, 100.0)
        return D

    elif pattern == 'layered':
        n_layers = rng.integers(2, 5)
        boundaries = np.sort(rng.uniform(0.1, 0.9, n_layers - 1))
        D = np.ones_like(X)
        for k in range(n_layers):
            lo = boundaries[k - 1] if k > 0 else 0.0
            hi = boundaries[k] if k < n_layers - 1 else 1.0
            mask = (X >= lo) & (X < hi)
            D[mask] = rng.uniform(0.1, 10.0)
        return D

    else:
        return np.ones_like(X)


def assemble_poisson_3d(N: int) -> sp.csr_matrix:
    """Assemble 3D Poisson matrix (-nabla^2 u = f) on [0,1]^3 with zero Dirichlet BCs.

    7-point stencil, N^3 interior DOFs.
    """
    T = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(N, N), format='csr')
    I = sp.eye(N, format='csr')
    I2 = sp.eye(N * N, format='csr')
    # A = I_N ⊗ I_N ⊗ T + I_N ⊗ T ⊗ I_N + T ⊗ I_N ⊗ I_N
    A = (sp.kron(I2, T, format='csr')
         + sp.kron(sp.kron(I, T, format='csr'), I, format='csr')
         + sp.kron(T, I2, format='csr'))
    return A


def get_grid_points_3d(N: int):
    """Return (X, Y, Z) meshgrid for N interior points per axis on [0,1]^3."""
    h = 1.0 / (N + 1)
    coords = np.linspace(h, 1.0 - h, N)
    return np.meshgrid(coords, coords, coords, indexing='ij')


def assemble_rhs_3d(f: np.ndarray, N: int) -> np.ndarray:
    """Scale 3D source term by h^2 and flatten."""
    h = 1.0 / (N + 1)
    return (h ** 2) * f.ravel()


def validate_matrix(A, N, dim: int = 2):
    n = N ** dim
    assert A.shape == (n, n)
    diff = A - A.T
    assert diff.nnz == 0 or abs(diff).max() < 1e-14
    max_nnz_per_row = np.diff(A.indptr).max()
    expected_max = 2 * dim + 1  # 5 for 2D, 7 for 3D
    assert max_nnz_per_row <= expected_max
    return True
