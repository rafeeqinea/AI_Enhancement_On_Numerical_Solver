import numpy as np
import pytest
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.solvers.cg import conjugate_gradient
from src.solvers.direct import solve_direct
from src.utils.metrics import compute_error, compute_speedup, summarize_run, summarize_experiment


class TestComputeError:
    def test_identical_is_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_error(x, x) == 0.0

    def test_known_error(self):
        ref = np.array([1.0, 0.0, 0.0])
        approx = np.array([0.9, 0.0, 0.0])
        assert abs(compute_error(approx, ref) - 0.1) < 1e-14

    def test_zero_reference(self):
        ref = np.zeros(5)
        approx = np.ones(5)
        assert compute_error(approx, ref) == 0.0


class TestComputeSpeedup:
    def test_equal_times(self):
        assert compute_speedup(1.0, 1.0) == 1.0

    def test_faster(self):
        assert compute_speedup(10.0, 2.0) == 5.0

    def test_zero_test_time(self):
        assert compute_speedup(1.0, 0.0) == float('inf')


class TestSummarizeRun:
    def test_keys_present(self):
        N = 4
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        cg = conjugate_gradient(A, b)
        direct = solve_direct(A, b)
        summary = summarize_run(N, cg, direct)

        expected_keys = {
            'N', 'dof', 'cg_iterations', 'cg_converged',
            'cg_time', 'direct_time', 'relative_error',
            'speedup', 'final_residual',
        }
        assert set(summary.keys()) == expected_keys

    def test_dof_correct(self):
        N = 8
        A = assemble_poisson_2d(N)
        b = np.ones(64)
        cg = conjugate_gradient(A, b)
        direct = solve_direct(A, b)
        summary = summarize_run(N, cg, direct)
        assert summary['dof'] == 64


class TestSummarizeExperiment:
    def test_aggregation(self):
        runs = [
            {'N': 4, 'cg_time': 0.01, 'direct_time': 0.02,
             'cg_iterations': 10, 'cg_converged': True},
            {'N': 8, 'cg_time': 0.03, 'direct_time': 0.05,
             'cg_iterations': 30, 'cg_converged': True},
        ]
        summary = summarize_experiment(runs)
        assert summary['num_runs'] == 2
        assert summary['max_iterations'] == 30
        assert summary['all_converged'] is True
        assert abs(summary['total_cg_time'] - 0.04) < 1e-14
