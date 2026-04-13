import numpy as np
import pytest
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct


class TestConjugateGradient:
    def test_converges(self):
        A = assemble_poisson_2d(8)
        b = np.random.default_rng(42).standard_normal(64)
        result = conjugate_gradient(A, b)
        assert result.converged

    def test_matches_direct(self):
        N = 16
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        direct = solve_direct(A, b)
        cg = conjugate_gradient(A, b, tol=1e-10)

        error = np.linalg.norm(cg.solution - direct.solution) / np.linalg.norm(direct.solution)
        assert error < 1e-6

    def test_warm_start_fewer_iterations(self):
        N = 64
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        direct = solve_direct(A, b)
        x0 = direct.solution * 0.99

        cold = conjugate_gradient(A, b, tol=1e-6)
        warm = conjugate_gradient(A, b, x0=x0, tol=1e-6)

        assert warm.iterations < cold.iterations

    def test_residual_history_decreasing(self):
        A = assemble_poisson_2d(16)
        b = np.random.default_rng(42).standard_normal(256)
        result = conjugate_gradient(A, b)

        for i in range(1, len(result.residual_history)):
            assert result.residual_history[i] <= result.residual_history[i - 1] * 1.01

    def test_zero_rhs(self):
        A = assemble_poisson_2d(4)
        b = np.zeros(16)
        result = conjugate_gradient(A, b)
        assert result.iterations == 0
        assert result.converged

    def test_iteration_count_scales_with_N(self):
        iters = {}
        for N in [8, 16, 32]:
            A = assemble_poisson_2d(N)
            X, Y = get_grid_points(N)
            rng = np.random.default_rng(42)
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            result = conjugate_gradient(A, b, tol=1e-6)
            iters[N] = result.iterations

        assert iters[16] > iters[8]
        assert iters[32] > iters[16]

    def test_timing_recorded(self):
        A = assemble_poisson_2d(8)
        b = np.ones(64)
        result = conjugate_gradient(A, b)
        assert result.time_seconds > 0

    def test_max_iter_stops(self):
        A = assemble_poisson_2d(32)
        b = np.random.default_rng(42).standard_normal(1024)
        result = conjugate_gradient(A, b, tol=1e-15, max_iter=5)
        assert result.iterations == 5
        assert not result.converged
