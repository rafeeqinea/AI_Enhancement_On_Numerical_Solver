import numpy as np
import pytest
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points, validate_matrix


class TestAssemblePoisson2D:
    def test_shape(self):
        A = assemble_poisson_2d(4)
        assert A.shape == (16, 16)

    def test_symmetric(self):
        A = assemble_poisson_2d(8)
        diff = A - A.T
        assert diff.nnz == 0

    def test_diagonal_is_four(self):
        A = assemble_poisson_2d(16)
        assert np.allclose(A.diagonal(), 4.0)

    def test_max_five_nonzeros_per_row(self):
        A = assemble_poisson_2d(8)
        nnz_per_row = np.diff(A.indptr)
        assert nnz_per_row.max() <= 5

    def test_positive_definite(self):
        A = assemble_poisson_2d(4)
        eigenvalues = np.linalg.eigvalsh(A.toarray())
        assert np.all(eigenvalues > 0)

    def test_known_eigenvalues(self):
        N = 4
        A = assemble_poisson_2d(N)
        h = 1.0 / (N + 1)
        computed = np.sort(np.linalg.eigvalsh(A.toarray()))
        expected = []
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                lam = 4 * np.sin(i * np.pi * h / 2)**2 + 4 * np.sin(j * np.pi * h / 2)**2
                expected.append(lam)
        expected = np.sort(expected)
        assert np.allclose(computed, expected, rtol=1e-10)

    def test_multiple_sizes(self):
        for N in [4, 8, 16, 32]:
            A = assemble_poisson_2d(N)
            assert A.shape == (N * N, N * N)


class TestAssembleRhs:
    def test_shape(self):
        f = np.ones((8, 8))
        b = assemble_rhs(f, 8)
        assert b.shape == (64,)

    def test_scaling(self):
        N = 4
        h = 1.0 / (N + 1)
        f = np.ones((N, N))
        b = assemble_rhs(f, N)
        assert np.allclose(b, h**2)


class TestGetGridPoints:
    def test_shape(self):
        X, Y = get_grid_points(8)
        assert X.shape == (8, 8)
        assert Y.shape == (8, 8)

    def test_interior_only(self):
        N = 4
        X, Y = get_grid_points(N)
        h = 1.0 / (N + 1)
        assert X.min() == pytest.approx(h)
        assert X.max() == pytest.approx(1.0 - h)


class TestValidateMatrix:
    def test_valid_matrix_passes(self):
        A = assemble_poisson_2d(8)
        assert validate_matrix(A, 8) is True
