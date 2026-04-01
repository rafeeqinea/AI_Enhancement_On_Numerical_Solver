from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

Preconditioner = Callable[[np.ndarray], np.ndarray]


def jacobi_preconditioner(A: sp.spmatrix) -> Preconditioner:
    diag = A.diagonal().copy()
    diag[diag == 0] = 1.0
    inv_diag = 1.0 / diag

    def apply(r: np.ndarray) -> np.ndarray:
        return inv_diag * r

    return apply


def ic0_preconditioner(A: sp.spmatrix, grid_N: int | None = None, dim: int = 2) -> Preconditioner:
    """Incomplete Cholesky IC(0) preconditioner.

    For small matrices (n < 5000): dense factorization.
    For large matrices with known grid structure: diagonal-based factorization.
    For large unstructured matrices: sparse CSC factorization.
    """
    n = A.shape[0]
    if n < 5000:
        return _ic0_dense(A)
    if grid_N is not None and grid_N > 2:
        return _ic0_structured(A, grid_N, dim)
    return _ic0_sparse_csc(A)


def _ic0_dense(A: sp.spmatrix) -> Preconditioner:
    """Dense IC(0) — fast for small matrices, O(n^2) memory."""
    A_csc = A.tocsc().astype(np.float64)
    n = A_csc.shape[0]
    L_data = A_csc.toarray()

    for k in range(n):
        L_data[k, k] = np.sqrt(max(L_data[k, k], 1e-14))
        rows = A_csc[:, k].nonzero()[0]
        rows = rows[rows > k]
        for i in rows:
            L_data[i, k] /= L_data[k, k]
        for i in rows:
            cols = A_csc[i, :].nonzero()[1]
            cols = cols[(cols > k) & (cols <= i)]
            for j in cols:
                L_data[i, j] -= L_data[i, k] * L_data[j, k]

    L = np.tril(L_data)
    L_sparse = sp.csr_matrix(L)
    Lt_sparse = sp.csc_matrix(L.T)

    def apply(r: np.ndarray) -> np.ndarray:
        y = spla.spsolve_triangular(L_sparse, r, lower=True)
        return spla.spsolve_triangular(Lt_sparse, y, lower=False)

    return apply


def _ic0_structured(A: sp.spmatrix, N: int, dim: int = 2) -> Preconditioner:
    """IC(0) for structured grid Poisson matrices using diagonal storage.

    Exploits the known stencil structure: 5-point (2D) or 7-point (3D).
    For N > 2, there are no cross-terms between sub-diagonals, making
    the factorization O(n) with tiny constant.
    """
    n = N ** dim
    assert A.shape[0] == n

    d = A.diagonal(0).copy().astype(np.float64)

    if dim == 2:
        # Sub-diagonals at offsets -1 and -N
        e1 = np.zeros(n, dtype=np.float64)
        eN = np.zeros(n, dtype=np.float64)

        diag_m1 = A.diagonal(-1).astype(np.float64)
        diag_mN = A.diagonal(-N).astype(np.float64)
        e1[1:1 + len(diag_m1)] = diag_m1
        eN[N:N + len(diag_mN)] = diag_mN

        for k in range(n):
            if k >= 1 and k % N != 0:
                d[k] -= e1[k] ** 2
            if k >= N:
                d[k] -= eN[k] ** 2
            d[k] = np.sqrt(max(d[k], 1e-14))

            if k + 1 < n and (k + 1) % N != 0:
                e1[k + 1] /= d[k]
            if k + N < n:
                eN[k + N] /= d[k]

        # Build sparse L from diagonals
        diags = [d]
        offsets = [0]
        if n > 1:
            diags.append(e1[1:])
            offsets.append(-1)
        if n > N:
            diags.append(eN[N:])
            offsets.append(-N)

    else:  # dim == 3
        N2 = N * N
        e1 = np.zeros(n, dtype=np.float64)
        eN = np.zeros(n, dtype=np.float64)
        eN2 = np.zeros(n, dtype=np.float64)

        diag_m1 = A.diagonal(-1).astype(np.float64)
        diag_mN = A.diagonal(-N).astype(np.float64)
        diag_mN2 = A.diagonal(-N2).astype(np.float64)
        e1[1:1 + len(diag_m1)] = diag_m1
        eN[N:N + len(diag_mN)] = diag_mN
        eN2[N2:N2 + len(diag_mN2)] = diag_mN2

        for k in range(n):
            if k >= 1 and k % N != 0:
                d[k] -= e1[k] ** 2
            if k >= N:
                d[k] -= eN[k] ** 2
            if k >= N2:
                d[k] -= eN2[k] ** 2
            d[k] = np.sqrt(max(d[k], 1e-14))

            if k + 1 < n and (k + 1) % N != 0:
                e1[k + 1] /= d[k]
            if k + N < n:
                eN[k + N] /= d[k]
            if k + N2 < n:
                eN2[k + N2] /= d[k]

        diags = [d]
        offsets = [0]
        if n > 1:
            diags.append(e1[1:])
            offsets.append(-1)
        if n > N:
            diags.append(eN[N:])
            offsets.append(-N)
        if n > N2:
            diags.append(eN2[N2:])
            offsets.append(-N2)

    L = sp.diags(diags, offsets, shape=(n, n), format='csr')
    Lt = sp.diags(diags, [-o for o in offsets], shape=(n, n), format='csc')

    def apply(r: np.ndarray) -> np.ndarray:
        y = spla.spsolve_triangular(L, r.astype(np.float64), lower=True)
        return spla.spsolve_triangular(Lt, y, lower=False)

    return apply


def _ic0_sparse_csc(A: sp.spmatrix) -> Preconditioner:
    """Generic sparse IC(0) for unstructured matrices. Slow but correct."""
    A_csc = sp.tril(A, format='csc').astype(np.float64)
    n = A_csc.shape[0]
    L = sp.lil_matrix((n, n), dtype=np.float64)

    A_lil = A_csc.tolil()
    for i in range(n):
        for j in A_lil.rows[i]:
            if j <= i:
                L[i, j] = A_lil[i, j]

    for k in range(n):
        L[k, k] = np.sqrt(max(L[k, k], 1e-14))
        col_k_rows = [i for i in range(k + 1, n) if k in L.rows[i]]

        for i in col_k_rows:
            L[i, k] /= L[k, k]

        for i in col_k_rows:
            lik = L[i, k]
            for j in col_k_rows:
                if j > i:
                    break
                if j in L.rows[i]:
                    L[i, j] -= lik * L[j, k]

    L_csr = L.tocsr()
    Lt_csc = L.T.tocsc()

    def apply(r: np.ndarray) -> np.ndarray:
        y = spla.spsolve_triangular(L_csr, r.astype(np.float64), lower=True)
        return spla.spsolve_triangular(Lt_csc, y, lower=False)

    return apply
