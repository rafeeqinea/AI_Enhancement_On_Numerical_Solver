import numpy as np
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.solvers.direct import solve_direct


class TestSolveDirect:
    def test_returns_result(self):
        A = assemble_poisson_2d(4)
        b = np.ones(16) * 0.04
        result = solve_direct(A, b)
        assert hasattr(result, 'solution')
        assert hasattr(result, 'time_seconds')

    def test_solution_satisfies_equation(self):
        A = assemble_poisson_2d(8)
        b = np.random.default_rng(42).standard_normal(64)
        result = solve_direct(A, b)
        residual = np.linalg.norm(A @ result.solution - b)
        assert residual < 1e-10

    def test_analytical_solution(self):
        N = 16
        X, Y = get_grid_points(N)
        f = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
        u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

        A = assemble_poisson_2d(N)
        b = assemble_rhs(f, N)
        result = solve_direct(A, b)

        error = np.linalg.norm(result.solution - u_exact.ravel()) / np.linalg.norm(u_exact.ravel())
        assert error < 1e-2

    def test_timing_positive(self):
        A = assemble_poisson_2d(4)
        b = np.ones(16)
        result = solve_direct(A, b)
        assert result.time_seconds >= 0
