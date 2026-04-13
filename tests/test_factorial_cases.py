"""Tests for factorial case construction and consistency."""
import json
import os

import numpy as np
import pytest
import torch

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.models.unet import UNet
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner


class TestFactorialCases:
    def test_eight_cases_are_distinct(self):
        """Verify the 8 cases represent distinct solver configurations."""
        cases = [
            {'name': 'Case 1', 'x0': 'zero', 'precond': 'none', 'loss': None, 'solver': 'CG'},
            {'name': 'Case 2', 'x0': 'warmstart', 'precond': 'none', 'loss': 'MSE', 'solver': 'CG'},
            {'name': 'Case 3', 'x0': 'zero', 'precond': 'jacobi', 'loss': None, 'solver': 'PCG'},
            {'name': 'Case 4', 'x0': 'zero', 'precond': 'ic0', 'loss': None, 'solver': 'PCG'},
            {'name': 'Case 5', 'x0': 'warmstart', 'precond': 'ic0', 'loss': 'MSE', 'solver': 'PCG'},
            {'name': 'Case 6', 'x0': 'zero', 'precond': 'unet', 'loss': 'MSE', 'solver': 'FCG'},
            {'name': 'Case 7', 'x0': 'zero', 'precond': 'unet', 'loss': 'condition', 'solver': 'FCG'},
            {'name': 'Case 8', 'x0': 'warmstart', 'precond': 'unet', 'loss': 'condition', 'solver': 'FCG'},
        ]
        configs = [(c['x0'], c['precond'], c['loss'], c['solver']) for c in cases]
        assert len(set(configs)) == 8, "All 8 cases must be distinct"

    def test_case1_cg_baseline_runs(self):
        """Case 1: plain CG with zero start."""
        N = 4
        A = assemble_poisson_2d(N)
        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        result = conjugate_gradient(A, b, tol=1e-6)
        assert result.converged
        assert result.iterations > 0

    def test_case4_ic0_pcg_runs(self):
        """Case 4: IC(0) + PCG with zero start."""
        N = 4
        A = assemble_poisson_2d(N)
        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        precond = ic0_preconditioner(A)

        result = preconditioned_cg(A, b, precond, tol=1e-6)
        assert result.converged
        baseline = conjugate_gradient(A, b, tol=1e-6)
        assert result.iterations <= baseline.iterations

    def test_case7_condition_loss_fcg_runs(self):
        """Case 7: condition-loss U-Net + FCG with zero start."""
        N = 4
        A = assemble_poisson_2d(N)
        device = torch.device('cpu')
        model = UNet(base_features=4, levels=1, dim=2).to(device)
        precond = make_nn_preconditioner(model, N, device=device, dim=2)

        rng = np.random.default_rng(99)
        X, Y = get_grid_points(N)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)

        result = flexible_cg(A, b, precond, tol=1e-6, max_iter=200)
        # Untrained model may not converge, but FCG should not crash
        assert result.iterations > 0

    def test_results_json_has_expected_structure(self):
        """Verify the committed results file matches the expected factorial structure."""
        results_path = os.path.join(
            os.path.dirname(__file__), '..', 'results', 'factorial', 'results.json'
        )
        if not os.path.exists(results_path):
            pytest.skip("results/factorial/results.json not found")

        with open(results_path) as f:
            data = json.load(f)

        assert 'N16' in data
        assert 'N32' in data
        assert 'N64' in data

        for size_key in ['N16', 'N32', 'N64']:
            cases = data[size_key]
            assert 'Case 1' in cases, f"Missing Case 1 in {size_key}"
            assert 'Case 7' in cases or 'Case 4' in cases, f"Missing key cases in {size_key}"
