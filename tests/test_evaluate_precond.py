"""Smoke and sanity tests for the evaluation pipeline."""
import numpy as np
import pytest

from src.data.poisson import assemble_poisson_2d
from src.evaluation.evaluate_precond import evaluate_preconditioner
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner


class TestEvaluatePreconditioner:
    def test_returns_expected_keys(self):
        A = assemble_poisson_2d(4)
        precond = jacobi_preconditioner(A)
        result = evaluate_preconditioner('jacobi', precond, N=4, num_samples=2, seed=99)
        expected_keys = {
            'precond_name', 'N', 'num_samples',
            'cold_iters_mean', 'cold_iters_std',
            'precond_iters_mean', 'precond_iters_std',
            'iteration_reduction', 'cold_time_mean', 'precond_time_mean',
            'speedup', 'mean_error', 'max_error',
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_finite(self):
        A = assemble_poisson_2d(4)
        precond = jacobi_preconditioner(A)
        result = evaluate_preconditioner('jacobi', precond, N=4, num_samples=2, seed=99)
        for key, val in result.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_ic0_reduces_iterations(self):
        A = assemble_poisson_2d(8)
        precond = ic0_preconditioner(A)
        result = evaluate_preconditioner('ic0', precond, N=8, num_samples=5, seed=99)
        assert result['iteration_reduction'] > 0, "IC(0) should reduce iterations"

    def test_fcg_path_runs(self):
        A = assemble_poisson_2d(4)
        precond = jacobi_preconditioner(A)
        result = evaluate_preconditioner(
            'jacobi', precond, N=4, num_samples=2, seed=99, use_fcg=True
        )
        assert result['precond_iters_mean'] > 0
