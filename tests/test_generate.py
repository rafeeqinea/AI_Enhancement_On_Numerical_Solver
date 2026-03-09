import os
import numpy as np
import pytest
from src.data.generate import generate_source_term, generate_dataset
from src.data.poisson import get_grid_points


class TestGenerateSourceTerm:
    def test_shape(self):
        X, Y = get_grid_points(8)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        assert f.shape == (8, 8)

    def test_positive_values(self):
        X, Y = get_grid_points(16)
        rng = np.random.default_rng(42)
        f = generate_source_term(X, Y, rng)
        assert np.all(f >= 0)

    def test_reproducible(self):
        X, Y = get_grid_points(8)
        f1 = generate_source_term(X, Y, np.random.default_rng(42))
        f2 = generate_source_term(X, Y, np.random.default_rng(42))
        assert np.allclose(f1, f2)

    def test_different_seeds_differ(self):
        X, Y = get_grid_points(8)
        f1 = generate_source_term(X, Y, np.random.default_rng(42))
        f2 = generate_source_term(X, Y, np.random.default_rng(99))
        assert not np.allclose(f1, f2)


class TestGenerateDataset:
    def test_creates_files(self, tmp_path):
        folder = generate_dataset(4, 10, seed=42, base_dir=str(tmp_path))
        assert os.path.exists(os.path.join(folder, 'sources.npy'))
        assert os.path.exists(os.path.join(folder, 'solutions.npy'))

    def test_correct_shapes(self, tmp_path):
        folder = generate_dataset(8, 20, seed=42, base_dir=str(tmp_path))
        sources = np.load(os.path.join(folder, 'sources.npy'))
        solutions = np.load(os.path.join(folder, 'solutions.npy'))
        assert sources.shape == (20, 8, 8)
        assert solutions.shape == (20, 8, 8)

    def test_solutions_satisfy_equation(self, tmp_path):
        N = 4
        from src.data.poisson import assemble_poisson_2d, assemble_rhs
        folder = generate_dataset(N, 5, seed=42, base_dir=str(tmp_path))
        sources = np.load(os.path.join(folder, 'sources.npy'))
        solutions = np.load(os.path.join(folder, 'solutions.npy'))
        A = assemble_poisson_2d(N)

        for i in range(5):
            b = assemble_rhs(sources[i], N)
            residual = np.linalg.norm(A @ solutions[i].ravel() - b)
            assert residual < 1e-10
