import os

import numpy as np

from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.solvers.direct import solve_direct


def generate_source_term(X, Y, rng, num_blobs=None):
    if num_blobs is None:
        num_blobs = rng.integers(3, 8)

    N = X.shape[0]
    f = np.zeros((N, N))

    for _ in range(num_blobs):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        amp = rng.uniform(0.5, 5.0)
        sigma = rng.uniform(0.05, 0.3)
        f += amp * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    return f


def generate_dataset(N, num_samples, seed, base_dir='data/processed'):
    rng = np.random.default_rng(seed)
    folder = os.path.join(base_dir, f'N{N}_{num_samples}samples_seed{seed}')
    os.makedirs(folder, exist_ok=True)

    A = assemble_poisson_2d(N)
    X, Y = get_grid_points(N)

    sources = np.zeros((num_samples, N, N))
    solutions = np.zeros((num_samples, N, N))

    for i in range(num_samples):
        f = generate_source_term(X, Y, rng)
        b = assemble_rhs(f, N)
        result = solve_direct(A, b)
        sources[i] = f
        solutions[i] = result.solution.reshape(N, N)

    np.save(os.path.join(folder, 'sources.npy'), sources)
    np.save(os.path.join(folder, 'solutions.npy'), solutions)
    return folder
