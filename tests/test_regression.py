"""Regression tests: verify current code reproduces committed results."""
import json
import os

import numpy as np
import pytest

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'factorial', 'results.json'
)


@pytest.fixture
def committed_results():
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("results/factorial/results.json not found")
    with open(RESULTS_PATH) as f:
        return json.load(f)


class TestRegression:
    def _run_case1_n16(self):
        """Reproduce Case 1 (plain CG) at N=16 with seed 99."""
        N = 16
        A = assemble_poisson_2d(N)
        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)

        iters = []
        for _ in range(50):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            result = conjugate_gradient(A, b, tol=1e-6)
            iters.append(result.iterations)

        return float(np.mean(iters))

    def test_case1_n16_matches_committed(self, committed_results):
        """Case 1 at N=16: reproduced mean iters must match committed results."""
        reproduced = self._run_case1_n16()
        committed = committed_results['N16']['Case 1']['mean_iters']
        assert abs(reproduced - committed) < 0.5, (
            f"Regression failure: reproduced {reproduced:.1f} vs committed {committed:.1f}"
        )

    def test_case4_n16_matches_committed(self, committed_results):
        """Case 4 (IC(0)+PCG) at N=16: reproduced mean iters must match committed results."""
        N = 16
        A = assemble_poisson_2d(N)
        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)
        precond = ic0_preconditioner(A)

        iters = []
        for _ in range(50):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            result = preconditioned_cg(A, b, precond, tol=1e-6)
            iters.append(result.iterations)

        reproduced = float(np.mean(iters))
        committed = committed_results['N16']['Case 4']['mean_iters']
        assert abs(reproduced - committed) < 0.5, (
            f"Regression failure: reproduced {reproduced:.1f} vs committed {committed:.1f}"
        )

    def test_case3_n16_matches_committed(self, committed_results):
        """Case 3 (Jacobi+PCG) at N=16: reproduced mean iters must match committed results."""
        N = 16
        A = assemble_poisson_2d(N)
        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)
        precond = jacobi_preconditioner(A)

        iters = []
        for _ in range(50):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)
            result = preconditioned_cg(A, b, precond, tol=1e-6)
            iters.append(result.iterations)

        reproduced = float(np.mean(iters))
        committed = committed_results['N16']['Case 3']['mean_iters']
        assert abs(reproduced - committed) < 0.5, (
            f"Regression failure: reproduced {reproduced:.1f} vs committed {committed:.1f}"
        )
