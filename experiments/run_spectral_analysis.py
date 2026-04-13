"""Spectral analysis: eigenvalue distributions of the preconditioned operator.

Constructs the dense preconditioned operator AM by applying each trained
preconditioner to all basis vectors at N=16, then computing eigenvalues.
This provides mechanistic evidence for why condition-loss works and MSE fails.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.poisson import assemble_poisson_2d
from src.models.unet import UNet
from src.solvers.preconditioners import ic0_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'spectral_analysis')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'report_figures')
N = 16
DOF = N * N  # 256


def build_preconditioner_matrix(precond_fn, n_dof: int) -> np.ndarray:
    """Build the dense matrix M by applying preconditioner to each basis vector."""
    M = np.zeros((n_dof, n_dof))
    for i in range(n_dof):
        e_i = np.zeros(n_dof)
        e_i[i] = 1.0
        M[:, i] = precond_fn(e_i)
    return M


def compute_spectrum(A_dense: np.ndarray, M: np.ndarray) -> dict:
    """Compute eigenvalues of the preconditioned operator A @ M."""
    AM = A_dense @ M
    eigenvalues = np.linalg.eigvals(AM)

    # Summary statistics
    real_parts = eigenvalues.real
    distances_from_one = np.abs(eigenvalues - 1.0)

    return {
        'eigenvalues_real': real_parts.tolist(),
        'eigenvalues_imag': eigenvalues.imag.tolist(),
        'mean_real': float(np.mean(real_parts)),
        'std_real': float(np.std(real_parts)),
        'min_real': float(np.min(real_parts)),
        'max_real': float(np.max(real_parts)),
        'mean_distance_from_1': float(np.mean(distances_from_one)),
        'max_distance_from_1': float(np.max(distances_from_one)),
        'condition_surrogate': float(np.max(real_parts) / max(np.min(real_parts), 1e-12)),
    }


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Grid: N={N}, DOF={DOF}')

    # Assemble system
    A_sparse = assemble_poisson_2d(N)
    A_dense = A_sparse.toarray()
    print(f'A shape: {A_dense.shape}')

    # 1. Unpreconditioned spectrum (eigenvalues of A)
    print('\n1. Computing spectrum of A (unpreconditioned)...')
    eig_A = np.linalg.eigvals(A_dense)
    unprecond = {
        'eigenvalues_real': eig_A.real.tolist(),
        'mean_real': float(np.mean(eig_A.real)),
        'min_real': float(np.min(eig_A.real)),
        'max_real': float(np.max(eig_A.real)),
        'condition_number': float(np.max(eig_A.real) / np.min(eig_A.real)),
    }
    print(f'   Condition number: {unprecond["condition_number"]:.1f}')
    print(f'   Eigenvalue range: [{unprecond["min_real"]:.3f}, {unprecond["max_real"]:.3f}]')

    # 2. IC(0) preconditioned spectrum
    print('\n2. Computing IC(0) preconditioned spectrum...')
    ic0_fn = ic0_preconditioner(A_sparse)
    M_ic0 = build_preconditioner_matrix(ic0_fn, DOF)
    ic0_spectrum = compute_spectrum(A_dense, M_ic0)
    print(f'   Mean distance from 1: {ic0_spectrum["mean_distance_from_1"]:.4f}')
    print(f'   Eigenvalue range: [{ic0_spectrum["min_real"]:.3f}, {ic0_spectrum["max_real"]:.3f}]')

    # 3. MSE-trained NN preconditioned spectrum
    print('\n3. Computing MSE-trained NN spectrum...')
    mse_model = UNet(base_features=16, levels=3)
    mse_ckpt = os.path.join(RESULTS_DIR, 'nn_precond',
                            'mse_checkpoints_N16', 'best_model.pt')
    mse_model.load_state_dict(
        torch.load(mse_ckpt, map_location=device, weights_only=True))
    mse_model = mse_model.to(device).eval()
    mse_fn = make_nn_preconditioner(mse_model, N, device=device)
    M_mse = build_preconditioner_matrix(mse_fn, DOF)
    mse_spectrum = compute_spectrum(A_dense, M_mse)
    print(f'   Mean distance from 1: {mse_spectrum["mean_distance_from_1"]:.4f}')
    print(f'   Eigenvalue range: [{mse_spectrum["min_real"]:.3f}, {mse_spectrum["max_real"]:.3f}]')

    # 4. Condition-loss-trained NN preconditioned spectrum
    print('\n4. Computing condition-loss NN spectrum...')
    cond_model = UNet(base_features=16, levels=3)
    cond_ckpt = os.path.join(RESULTS_DIR, 'nn_precond',
                             'condition_checkpoints_N16', 'best_model.pt')
    cond_model.load_state_dict(
        torch.load(cond_ckpt, map_location=device, weights_only=True))
    cond_model = cond_model.to(device).eval()
    cond_fn = make_nn_preconditioner(cond_model, N, device=device)
    M_cond = build_preconditioner_matrix(cond_fn, DOF)
    cond_spectrum = compute_spectrum(A_dense, M_cond)
    print(f'   Mean distance from 1: {cond_spectrum["mean_distance_from_1"]:.4f}')
    print(f'   Eigenvalue range: [{cond_spectrum["min_real"]:.3f}, {cond_spectrum["max_real"]:.3f}]')

    # Save results
    results = {
        'description': 'Eigenvalue spectrum of preconditioned operator AM at N=16',
        'grid_size': N,
        'dof': DOF,
        'unpreconditioned': unprecond,
        'ic0': ic0_spectrum,
        'mse_nn': mse_spectrum,
        'condition_nn': cond_spectrum,
    }

    json_path = os.path.join(OUTPUT_DIR, 'spectrum_N16.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {json_path}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    configs = [
        (axes[0, 0], eig_A.real, 'Unpreconditioned A',
         f'Eigenvalues of A\n(condition number = {unprecond["condition_number"]:.0f})',
         '#666666'),
        (axes[0, 1], ic0_spectrum['eigenvalues_real'],
         'IC(0) Preconditioned',
         f'Eigenvalues of AM (IC(0))\nmean |1-lambda| = {ic0_spectrum["mean_distance_from_1"]:.3f}',
         '#70AD47'),
        (axes[1, 0], mse_spectrum['eigenvalues_real'],
         'MSE-Trained NN',
         f'Eigenvalues of AM (MSE U-Net)\nmean |1-lambda| = {mse_spectrum["mean_distance_from_1"]:.3f}',
         '#E74C3C'),
        (axes[1, 1], cond_spectrum['eigenvalues_real'],
         'Condition-Loss NN',
         f'Eigenvalues of AM (Condition-Loss U-Net)\nmean |1-lambda| = {cond_spectrum["mean_distance_from_1"]:.3f}',
         '#ED7D31'),
    ]

    for ax, eigs, label, title, color in configs:
        eigs = np.array(eigs)
        ax.hist(eigs, bins=40, color=color, alpha=0.75, edgecolor='#333',
                linewidth=0.5)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1,
                   label='Ideal (lambda=1)')
        ax.set_xlabel('Eigenvalue (real part)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Eigenvalue Spectra of Preconditioned Operators at N=16\n'
                 '(Good preconditioner = eigenvalues clustered near 1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    fig_path = os.path.join(FIGURES_DIR, 'fig_spectrum_n16.png')
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {fig_path}')

    # Summary
    print('\n' + '=' * 60)
    print('SPECTRAL ANALYSIS SUMMARY')
    print('=' * 60)
    print(f'{"Method":<25} {"Mean |1-eig|":>12} {"eig range":>20}')
    print('-' * 57)
    print(f'{"Unprecond (A)":<25} {"n/a":>12} '
          f'[{unprecond["min_real"]:.2f}, {unprecond["max_real"]:.2f}]')
    print(f'{"IC(0)":<25} {ic0_spectrum["mean_distance_from_1"]:>12.4f} '
          f'[{ic0_spectrum["min_real"]:.3f}, {ic0_spectrum["max_real"]:.3f}]')
    print(f'{"MSE U-Net":<25} {mse_spectrum["mean_distance_from_1"]:>12.4f} '
          f'[{mse_spectrum["min_real"]:.3f}, {mse_spectrum["max_real"]:.3f}]')
    print(f'{"Condition-Loss U-Net":<25} {cond_spectrum["mean_distance_from_1"]:>12.4f} '
          f'[{cond_spectrum["min_real"]:.3f}, {cond_spectrum["max_real"]:.3f}]')


if __name__ == '__main__':
    run()
