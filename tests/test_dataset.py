import numpy as np
import pytest
import torch
from src.data.dataset import PoissonDataset
from src.data.generate import generate_dataset


@pytest.fixture
def dataset_dir(tmp_path):
    return generate_dataset(8, 20, seed=42, base_dir=str(tmp_path))


class TestPoissonDataset:
    def test_length(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        assert len(ds) == 20

    def test_output_types(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_input_shape_padded(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        x, y = ds[0]
        assert x.shape == (1, 10, 10)
        assert y.shape == (1, 8, 8)

    def test_boundary_is_zero(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        x, _ = ds[0]
        assert torch.all(x[0, 0, :] == 0)
        assert torch.all(x[0, -1, :] == 0)
        assert torch.all(x[0, :, 0] == 0)
        assert torch.all(x[0, :, -1] == 0)

    def test_interior_matches_source(self, dataset_dir):
        ds = PoissonDataset(dataset_dir, normalise=False)
        x, _ = ds[0]
        raw_source = np.load(dataset_dir + '/sources.npy')[0]
        interior = x[0, 1:-1, 1:-1].numpy()
        assert np.allclose(interior, raw_source, atol=1e-6)

    def test_normalisation(self, dataset_dir):
        ds = PoissonDataset(dataset_dir, normalise=True)
        x, y = ds[0]
        raw_source = np.load(dataset_dir + '/sources.npy')[0]
        expected = (raw_source - ds.source_mean) / ds.source_std
        interior = x[0, 1:-1, 1:-1].numpy()
        assert np.allclose(interior, expected, atol=1e-6)

    def test_dtype_float32(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_dataloader_compatible(self, dataset_dir):
        ds = PoissonDataset(dataset_dir)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 1, 10, 10)
        assert batch_y.shape == (4, 1, 8, 8)
