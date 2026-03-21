import numpy as np
import pytest

from src.data.poisson import (
    assemble_poisson_2d, assemble_variable_poisson_2d,
    get_grid_points, generate_diffusion_coefficient,
)
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct
from src.data.generate import generate_source_term


@pytest.fixture
def grid():
    N = 16
    X, Y = get_grid_points(N)
    return N, X, Y


class TestVariablePoissonAssembly:
    def test_constant_d_matches_standard(self, grid):
        N, X, Y = grid
        D = np.ones((N, N))
        A_var = assemble_variable_poisson_2d(N, D)
        A_std = assemble_poisson_2d(N)
        h = 1.0 / (N + 1)
        diff = (A_var * h**2 - A_std).toarray()
        assert np.max(np.abs(diff)) < 1e-12

    def test_symmetric(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        for pattern in ['smooth', 'discontinuous', 'layered']:
            D = generate_diffusion_coefficient(X, Y, rng, pattern=pattern)
            A = assemble_variable_poisson_2d(N, D)
            sym_err = (A - A.T).toarray()
            assert np.max(np.abs(sym_err)) < 1e-14

    def test_positive_definite(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        D = generate_diffusion_coefficient(X, Y, rng, 'discontinuous')
        A = assemble_variable_poisson_2d(N, D)
        eigenvalues = np.linalg.eigvalsh(A.toarray())
        assert eigenvalues.min() > 0

    def test_cg_converges(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        D = generate_diffusion_coefficient(X, Y, rng, 'smooth')
        A = assemble_variable_poisson_2d(N, D)
        f = generate_source_term(X, Y, rng)
        b = f.ravel()
        result = conjugate_gradient(A, b, tol=1e-6)
        assert result.converged

    def test_shape(self, grid):
        N, X, Y = grid
        D = np.ones((N, N)) * 5.0
        A = assemble_variable_poisson_2d(N, D)
        assert A.shape == (N * N, N * N)

    def test_higher_contrast_more_iterations(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)

        D_easy = np.ones((N, N))
        A_easy = assemble_variable_poisson_2d(N, D_easy)
        r_easy = conjugate_gradient(A_easy, f.ravel(), tol=1e-6)

        D_hard = np.ones((N, N))
        D_hard[:N//2, :] = 100.0
        A_hard = assemble_variable_poisson_2d(N, D_hard)
        r_hard = conjugate_gradient(A_hard, f.ravel(), tol=1e-6)

        assert r_hard.iterations > r_easy.iterations


class TestDiffusionCoefficient:
    def test_smooth_positive(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        D = generate_diffusion_coefficient(X, Y, rng, 'smooth')
        assert D.min() > 0
        assert D.shape == (N, N)

    def test_discontinuous_has_jump(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        D = generate_diffusion_coefficient(X, Y, rng, 'discontinuous')
        assert D.max() / D.min() > 5.0

    def test_layered_shape(self, grid):
        N, X, Y = grid
        rng = np.random.default_rng(42)
        D = generate_diffusion_coefficient(X, Y, rng, 'layered')
        assert D.shape == (N, N)
        assert D.min() > 0
