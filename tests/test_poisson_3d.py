import numpy as np
import pytest
import scipy.sparse as sp

from src.data.poisson import assemble_poisson_3d, assemble_rhs_3d, get_grid_points_3d, validate_matrix
from src.data.generate import generate_source_term_3d
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.solvers.direct import solve_direct


class TestPoisson3DAssembly:
    def test_shape(self):
        N = 4
        A = assemble_poisson_3d(N)
        assert A.shape == (64, 64)

    def test_symmetric(self):
        A = assemble_poisson_3d(8)
        diff = A - A.T
        assert diff.nnz == 0 or abs(diff).max() < 1e-14

    def test_spd(self):
        A = assemble_poisson_3d(4)
        eigvals = np.linalg.eigvalsh(A.toarray())
        assert eigvals.min() > 0

    def test_diagonal(self):
        A = assemble_poisson_3d(8)
        assert np.allclose(A.diagonal(), 6.0)

    def test_nnz_per_row(self):
        N = 8
        A = assemble_poisson_3d(N)
        nnz_per_row = np.diff(A.indptr)
        assert nnz_per_row.max() <= 7

    def test_validate(self):
        A = assemble_poisson_3d(8)
        assert validate_matrix(A, 8, dim=3)


class TestPoisson3DGrid:
    def test_grid_shape(self):
        X, Y, Z = get_grid_points_3d(8)
        assert X.shape == (8, 8, 8)
        assert Y.shape == (8, 8, 8)
        assert Z.shape == (8, 8, 8)

    def test_grid_range(self):
        X, Y, Z = get_grid_points_3d(8)
        for G in [X, Y, Z]:
            assert G.min() > 0
            assert G.max() < 1


class TestPoisson3DSource:
    def test_source_shape(self):
        X, Y, Z = get_grid_points_3d(8)
        rng = np.random.default_rng(42)
        f = generate_source_term_3d(X, Y, Z, rng)
        assert f.shape == (8, 8, 8)

    def test_rhs_shape(self):
        f = np.ones((8, 8, 8))
        b = assemble_rhs_3d(f, 8)
        assert b.shape == (512,)


class TestPoisson3DSolvers:
    @pytest.fixture
    def system_3d(self):
        N = 8
        A = assemble_poisson_3d(N)
        X, Y, Z = get_grid_points_3d(N)
        rng = np.random.default_rng(42)
        f = generate_source_term_3d(X, Y, Z, rng)
        b = assemble_rhs_3d(f, N)
        return A, b, N

    def test_cg_converges(self, system_3d):
        A, b, N = system_3d
        result = conjugate_gradient(A, b, tol=1e-6)
        assert result.converged
        assert result.iterations < 100

    def test_cg_matches_direct(self, system_3d):
        A, b, N = system_3d
        direct = solve_direct(A, b)
        cg_result = conjugate_gradient(A, b, tol=1e-10)
        assert np.allclose(cg_result.solution, direct.solution, atol=1e-6)

    def test_jacobi_pcg(self, system_3d):
        A, b, N = system_3d
        M = jacobi_preconditioner(A)
        result = preconditioned_cg(A, b, M, tol=1e-6)
        assert result.converged

    def test_ic0_pcg(self, system_3d):
        A, b, N = system_3d
        M = ic0_preconditioner(A, grid_N=N, dim=3)
        cg_result = conjugate_gradient(A, b, tol=1e-6)
        pcg_result = preconditioned_cg(A, b, M, tol=1e-6)
        assert pcg_result.converged
        assert pcg_result.iterations < cg_result.iterations

    def test_fcg(self, system_3d):
        A, b, N = system_3d
        M = jacobi_preconditioner(A)
        result = flexible_cg(A, b, M, tol=1e-6)
        assert result.converged


class TestUNet3D:
    def test_forward(self):
        import torch
        from src.models.unet import UNet
        model = UNet(base_features=16, levels=3, dim=3)
        x = torch.randn(1, 1, 18, 18, 18)  # N=16 → 18 with padding
        y = model(x)
        assert y.shape == (1, 1, 16, 16, 16)

    def test_output_shape_n32(self):
        import torch
        from src.models.unet import UNet
        model = UNet(base_features=16, levels=3, dim=3)
        x = torch.randn(1, 1, 34, 34, 34)
        y = model(x)
        assert y.shape == (1, 1, 32, 32, 32)


class TestConditionLoss3D:
    def test_loss_computes(self):
        import torch
        from src.models.unet import UNet
        from src.training.losses import ConditionLoss

        N = 4
        A = assemble_poisson_3d(N)
        device = torch.device('cpu')
        model = UNet(base_features=8, levels=2, dim=3)
        loss_fn = ConditionLoss(A, N, num_probes=4, dim=3)
        loss = loss_fn(model, device)
        assert loss.item() > 0
        loss.backward()

    def test_loss_with_optimizations(self):
        import torch
        from src.models.unet import UNet
        from src.training.losses import ConditionLoss

        N = 4
        A = assemble_poisson_3d(N)
        device = torch.device('cpu')
        model = UNet(base_features=8, levels=2, dim=3)
        loss_fn = ConditionLoss(A, N, num_probes=8, dim=3)
        loss = loss_fn(model, device, use_checkpointing=True, probe_batch_size=4)
        assert loss.item() > 0
        loss.backward()


class TestNNPreconditioner3D:
    def test_make_preconditioner(self):
        import torch
        from src.models.unet import UNet
        from src.evaluation.nn_preconditioner import make_nn_preconditioner

        N = 4
        model = UNet(base_features=8, levels=2, dim=3)
        precond = make_nn_preconditioner(model, N, dim=3)

        r = np.random.randn(N ** 3)
        z = precond(r)
        assert z.shape == r.shape

    def test_zero_input(self):
        import torch
        from src.models.unet import UNet
        from src.evaluation.nn_preconditioner import make_nn_preconditioner

        N = 4
        model = UNet(base_features=8, levels=2, dim=3)
        precond = make_nn_preconditioner(model, N, dim=3)
        z = precond(np.zeros(N ** 3))
        assert np.allclose(z, 0)
