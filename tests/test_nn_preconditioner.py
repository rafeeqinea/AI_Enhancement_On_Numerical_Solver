import numpy as np
import pytest
import torch

from src.models.unet import UNet
from src.evaluation.nn_preconditioner import make_nn_preconditioner
from src.solvers.fcg import flexible_cg
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term


@pytest.fixture
def small_unet():
    return UNet(base_features=4, levels=2)


class TestNNPreconditioner:
    def test_returns_callable(self, small_unet):
        precond = make_nn_preconditioner(small_unet, N=8)
        assert callable(precond)

    def test_output_shape(self, small_unet):
        precond = make_nn_preconditioner(small_unet, N=8)
        r = np.random.randn(64).astype(np.float32)
        z = precond(r)
        assert z.shape == (64,)

    def test_deterministic(self, small_unet):
        precond = make_nn_preconditioner(small_unet, N=8)
        r = np.random.randn(64).astype(np.float32)
        z1 = precond(r)
        z2 = precond(r)
        np.testing.assert_array_equal(z1, z2)

    def test_works_in_fcg(self, small_unet):
        N = 8
        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        precond = make_nn_preconditioner(small_unet, N=N)
        result = flexible_cg(A, b, precond, max_iter=500)
        assert result.iterations <= 500
