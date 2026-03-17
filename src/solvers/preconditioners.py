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


def ic0_preconditioner(A: sp.spmatrix) -> Preconditioner:
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
