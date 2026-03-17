import numpy as np
import pytest

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg


@pytest.fixture
def poisson_system():
    N = 16
    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)
    rng = np.random.default_rng(42)
    f = generate_source_term(X, Y, rng)
    b = assemble_rhs(f, N)
    return A, b, N


class TestJacobiPreconditioner:
    def test_diagonal_inverse(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        r = np.ones(N * N)
        z = precond(r)
        np.testing.assert_allclose(z, r / 4.0, rtol=1e-12)

    def test_output_shape(self, poisson_system):
        A, b, N = poisson_system
        precond = jacobi_preconditioner(A)
        z = precond(b)
        assert z.shape == b.shape

    def test_pcg_fewer_iterations(self, poisson_system):
        A, b, N = poisson_system
        cg_result = conjugate_gradient(A, b)
        precond = jacobi_preconditioner(A)
        pcg_result = preconditioned_cg(A, b, precond)
        assert pcg_result.converged
        assert pcg_result.iterations <= cg_result.iterations


class TestIC0Preconditioner:
    def test_output_shape(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        z = precond(b)
        assert z.shape == b.shape

    def test_approximate_solve(self, poisson_system):
        A, b, N = poisson_system
        precond = ic0_preconditioner(A)
        z = precond(b)
        from src.solvers.direct import solve_direct
        exact = solve_direct(A, b)
        rel_err = np.linalg.norm(z - exact.solution) / np.linalg.norm(exact.solution)
        assert rel_err < 1.0

    def test_pcg_fewer_iterations(self, poisson_system):
        A, b, N = poisson_system
        cg_result = conjugate_gradient(A, b)
        precond = ic0_preconditioner(A)
        pcg_result = preconditioned_cg(A, b, precond)
        assert pcg_result.converged
        assert pcg_result.iterations < cg_result.iterations

    def test_better_than_jacobi(self, poisson_system):
        A, b, N = poisson_system
        jac = jacobi_preconditioner(A)
        ic0 = ic0_preconditioner(A)
        jac_result = preconditioned_cg(A, b, jac)
        ic0_result = preconditioned_cg(A, b, ic0)
        assert ic0_result.iterations <= jac_result.iterations
