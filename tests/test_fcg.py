import numpy as np
import pytest

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import CGResult, conjugate_gradient
from src.solvers.fcg import flexible_cg
from src.solvers.pcg import preconditioned_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.solvers.direct import solve_direct


@pytest.fixture
def poisson_system():
    N = 16
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)
    rng = np.random.default_rng(42)
    f = generate_source_term(X, Y, rng)
    b = assemble_rhs(f, N)
    return A, b, N


class TestFlexibleCG:
    def test_converges_with_identity_precond(self, poisson_system):
        A, b, N = poisson_system
        identity = lambda r: r.copy()
        fcg_result = flexible_cg(A, b, identity)
        cg_result = conjugate_gradient(A, b)
        assert fcg_result.converged
        assert abs(fcg_result.iterations - cg_result.iterations) <= 2

    def test_converges_with_jacobi(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = flexible_cg(A, b, precond)
        assert result.converged

    def test_converges_with_ic0(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        result = flexible_cg(A, b, precond)
        assert result.converged

    def test_matches_direct_solution(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        fcg_result = flexible_cg(A, b, precond)
        direct_result = solve_direct(A, b)
        rel_err = np.linalg.norm(fcg_result.solution - direct_result.solution) / np.linalg.norm(direct_result.solution)
        assert rel_err < 1e-4

    def test_m_max_parameter(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result_m1 = flexible_cg(A, b, precond, m_max=1)
        result_m20 = flexible_cg(A, b, precond, m_max=20)
        assert result_m1.converged
        assert result_m20.converged

    def test_returns_cgresult(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = flexible_cg(A, b, precond)
        assert isinstance(result, CGResult)

    def test_nonlinear_precond(self, poisson_system):
        A, b, N = poisson_system
        diag = A.diagonal()

        def nonlinear_precond(r: np.ndarray) -> np.ndarray:
            return r / diag * (1.0 + 0.01 * np.sin(r))

        result = flexible_cg(A, b, nonlinear_precond, max_iter=5000)
        assert result.converged

    def test_warm_start_compatible(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        x0 = np.ones(N * N) * 0.01
        result = flexible_cg(A, b, precond, x0=x0)
        assert result.converged
