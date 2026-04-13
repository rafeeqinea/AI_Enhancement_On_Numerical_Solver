import numpy as np
import pytest

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import CGResult, conjugate_gradient
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


class TestPreconditionedCG:
    def test_converges_with_jacobi(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = preconditioned_cg(A, b, precond)
        assert result.converged
        assert result.iterations > 0

    def test_converges_with_ic0(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        result = preconditioned_cg(A, b, precond)
        assert result.converged

    def test_matches_direct_solution(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        pcg_result = preconditioned_cg(A, b, precond)
        direct_result = solve_direct(A, b)
        rel_err = np.linalg.norm(pcg_result.solution - direct_result.solution) / np.linalg.norm(direct_result.solution)
        assert rel_err < 1e-4

    def test_fewer_iterations_than_cg(self, poisson_system):
        A, b, N = poisson_system
        cg_result = conjugate_gradient(A, b)
        precond = ic0_preconditioner(A)
        pcg_result = preconditioned_cg(A, b, precond)
        assert pcg_result.iterations < cg_result.iterations

    def test_warm_start_compatible(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        x0 = np.ones(N * N) * 0.01
        result = preconditioned_cg(A, b, precond, x0=x0)
        assert result.converged

    def test_residual_history_recorded(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = preconditioned_cg(A, b, precond)
        assert len(result.residual_history) > 0
        assert result.residual_history[-1] < 1e-6

    def test_returns_cgresult(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = preconditioned_cg(A, b, precond)
        assert isinstance(result, CGResult)

    def test_timing_recorded(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        result = preconditioned_cg(A, b, precond)
        assert result.time_seconds > 0

    def test_ic0_fewer_than_jacobi(self, poisson_system):
        A, b, N = poisson_system
        jac = jacobi_preconditioner(A)
        ic0 = ic0_preconditioner(A)
        jac_result = preconditioned_cg(A, b, jac)
        ic0_result = preconditioned_cg(A, b, ic0)
        assert ic0_result.iterations <= jac_result.iterations
