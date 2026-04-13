import os
import numpy as np
import pytest
from src.solvers.cg import CGResult
from src.utils.visualize import plot_convergence, plot_solution, plot_scaling, plot_comparison_bar


@pytest.fixture
def sample_cg_result():
    return CGResult(
        solution=np.random.default_rng(42).standard_normal(64),
        iterations=50,
        converged=True,
        residual_history=[10 ** (-i / 20) for i in range(51)],
        time_seconds=0.05,
    )


class TestPlotConvergence:
    def test_returns_figure(self, sample_cg_result):
        fig = plot_convergence(sample_cg_result)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_file(self, sample_cg_result, tmp_path):
        path = tmp_path / 'conv.png'
        plot_convergence(sample_cg_result, save_path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0


class TestPlotSolution:
    def test_returns_figure(self):
        u = np.random.default_rng(42).standard_normal(64)
        fig = plot_solution(u, 8)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        u = np.random.default_rng(42).standard_normal(64)
        path = tmp_path / 'sol.png'
        plot_solution(u, 8, save_path=str(path))
        assert path.exists()


class TestPlotScaling:
    def test_returns_figure(self):
        fig = plot_scaling([4, 8, 16], [10, 30, 90])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        path = tmp_path / 'scaling.png'
        plot_scaling([4, 8, 16], [10, 30, 90], save_path=str(path))
        assert path.exists()


class TestPlotComparisonBar:
    def test_returns_figure(self):
        fig = plot_comparison_bar(['CG', 'Direct'], [0.1, 0.5])
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_saves_file(self, tmp_path):
        path = tmp_path / 'bar.png'
        plot_comparison_bar(['CG', 'Direct'], [0.1, 0.5], save_path=str(path))
        assert path.exists()
