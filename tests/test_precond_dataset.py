import numpy as np
import pytest
import tempfile
from pathlib import Path

from src.data.precond_dataset import generate_precond_data, PrecondDataset
from src.data.poisson import assemble_poisson_2d


@pytest.fixture
def precond_data_dir(tmp_path):
    return generate_precond_data(N=8, num_systems=5, cg_iters=10, seed=42, base_dir=tmp_path)


class TestPrecondDataGeneration:
    def test_output_files_exist(self, precond_data_dir):
        assert (precond_data_dir / 'residuals.npy').exists()
        assert (precond_data_dir / 'errors.npy').exists()

    def test_output_shapes(self, precond_data_dir):
        residuals = np.load(precond_data_dir / 'residuals.npy')
        errors = np.load(precond_data_dir / 'errors.npy')
        assert residuals.shape == (50, 8, 8)
        assert errors.shape == (50, 8, 8)

    def test_errors_satisfy_relation(self, precond_data_dir):
        residuals = np.load(precond_data_dir / 'residuals.npy')
        errors = np.load(precond_data_dir / 'errors.npy')
        A = assemble_poisson_2d(8)
        for i in range(min(5, len(residuals))):
            r = residuals[i].ravel()
            e = errors[i].ravel()
            Ae = A @ e
            rel = np.linalg.norm(Ae - r) / np.linalg.norm(r)
            assert rel < 1e-6


class TestPrecondDataset:
    def test_length(self, precond_data_dir):
        ds = PrecondDataset(precond_data_dir)
        assert len(ds) == 50

    def test_output_shapes(self, precond_data_dir):
        ds = PrecondDataset(precond_data_dir)
        x, y = ds[0]
        assert x.shape == (1, 10, 10)
        assert y.shape == (1, 8, 8)

    def test_normalisation(self, precond_data_dir):
        ds = PrecondDataset(precond_data_dir, normalise=True)
        assert ds.res_std > 0
        assert ds.err_std > 0

    def test_no_normalisation(self, precond_data_dir):
        ds = PrecondDataset(precond_data_dir, normalise=False)
        assert ds.res_mean == 0.0
        assert ds.res_std == 1.0
