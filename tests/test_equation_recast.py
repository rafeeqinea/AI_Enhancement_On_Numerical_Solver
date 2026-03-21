import numpy as np
import pytest

from src.data.poisson import (
    assemble_poisson_2d, assemble_variable_poisson_2d,
    get_grid_points, generate_diffusion_coefficient, assemble_rhs,
)
from src.data.generate import generate_source_term
from src.solvers.equation_recast import recast_solve
from src.solvers.preconditioners import ic0_preconditioner
from src.solvers.direct import solve_direct
from src.solvers.cg import conjugate_gradient


@pytest.fixture
def variable_system():
    N = 16
    X, Y = get_grid_points(N)
    rng = np.random.default_rng(42)
    D = np.ones((N, N)) * 1.02
    D[:N//2, :] = 1.1
    A_ref = assemble_variable_poisson_2d(N, np.ones((N, N)))
    A_new = assemble_variable_poisson_2d(N, D)
    f = generate_source_term(X, Y, rng)
    b = f.ravel()
    return A_ref, A_new, b, N


class TestEquationRecast:
    def test_converges_with_ic0(self, variable_system):
        A_ref, A_new, b, N = variable_system
        precond = ic0_preconditioner(A_ref)
        result = recast_solve(A_ref, A_new, b, precond, tol=1e-6)
        assert result.converged

    def test_matches_direct(self, variable_system):
        A_ref, A_new, b, N = variable_system
        precond = ic0_preconditioner(A_ref)
        recast = recast_solve(A_ref, A_new, b, precond, tol=1e-6)
        direct = solve_direct(A_new, b)
        rel_err = np.linalg.norm(recast.solution - direct.solution) / np.linalg.norm(direct.solution)
        assert rel_err < 1e-4

    def test_same_operator_converges_immediately(self):
        N = 8
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        precond = ic0_preconditioner(A)
        result = recast_solve(A, A, b, precond, tol=1e-6)
        assert result.converged
        assert len(result.residual_history) <= 2

    def test_returns_cgresult(self, variable_system):
        from src.solvers.cg import CGResult
        A_ref, A_new, b, N = variable_system
        precond = ic0_preconditioner(A_ref)
        result = recast_solve(A_ref, A_new, b, precond, tol=1e-6)
        assert isinstance(result, CGResult)

    def test_fewer_outer_with_small_perturbation(self):
        N = 8
        X, Y = get_grid_points(N)
        A_ref = assemble_variable_poisson_2d(N, np.ones((N, N)))

        D_small = np.ones((N, N)) * 1.05
        A_small = assemble_variable_poisson_2d(N, D_small)

        D_big = np.ones((N, N))
        D_big[:N//2, :] = 3.0
        A_big = assemble_variable_poisson_2d(N, D_big)

        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = f.ravel()
        precond = ic0_preconditioner(A_ref)

        r_small = recast_solve(A_ref, A_small, b, precond, tol=1e-6)
        r_big = recast_solve(A_ref, A_big, b, precond, tol=1e-6)

        assert len(r_small.residual_history) <= len(r_big.residual_history)
