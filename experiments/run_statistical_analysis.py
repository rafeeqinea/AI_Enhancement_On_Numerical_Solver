"""Statistical analysis for core factorial comparisons.

Reproduces the 50 per-problem iteration counts using seed 99 (same as
run_factorial.py), runs paired nonparametric tests, and saves results.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data.poisson import assemble_poisson_2d, assemble_rhs, get_grid_points
from src.data.generate import generate_source_term
from src.models.unet import UNet
from src.solvers.cg import conjugate_gradient
from src.solvers.pcg import preconditioned_cg
from src.solvers.fcg import flexible_cg
from src.solvers.preconditioners import jacobi_preconditioner, ic0_preconditioner
from src.evaluation.nn_preconditioner import make_nn_preconditioner

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
EVAL_SIZES = [16, 32, 64]
NUM_SAMPLES = 50
SEED = 99
TOL = 1e-6


def collect_paired_data() -> dict:
    """Run Cases 1, 2, 4, 7 on the same 50 problems per grid. Return raw iters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load condition-loss model loader
    def load_cond_model(N):
        model = UNet(base_features=16, levels=3)
        ckpt = os.path.join(RESULTS_DIR, 'nn_precond',
                            f'condition_checkpoints_N{N}', 'best_model.pt')
        if not os.path.exists(ckpt):
            return None
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        return model.to(device)

    # Load warm-start model
    from src.evaluation.evaluate import predict_warmstart
    from src.data.dataset import PoissonDataset
    ws_model = UNet(base_features=16, levels=2)
    ws_ckpt = os.path.join(RESULTS_DIR, 'warmstart', 'unet_checkpoints',
                           'best_model.pt')
    ws_stats = {'source_mean': 0.0, 'source_std': 1.0,
                'sol_mean': 0.0, 'sol_std': 1.0}
    if os.path.exists(ws_ckpt):
        ws_model.load_state_dict(
            torch.load(ws_ckpt, map_location=device, weights_only=True))
        ws_model = ws_model.to(device)
        # Load stats from first available dataset
        base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
        for N in EVAL_SIZES:
            d = os.path.join(base, f'N{N}_3000samples_seed42')
            if os.path.exists(d):
                ds = PoissonDataset(d, normalise=True)
                ws_stats = {
                    'source_mean': ds.source_mean, 'source_std': ds.source_std,
                    'sol_mean': ds.sol_mean, 'sol_std': ds.sol_std,
                }
                break
    else:
        ws_model = None

    all_data = {}

    for N in EVAL_SIZES:
        print(f'\n{"="*60}')
        print(f'Collecting paired data — N={N}')
        print(f'{"="*60}')

        A = assemble_poisson_2d(N)
        X, Y = get_grid_points(N)
        rng = np.random.default_rng(SEED)

        ic0_precond = ic0_preconditioner(A)

        cond_model = load_cond_model(N)
        nn_precond = None
        if cond_model is not None:
            nn_precond = make_nn_preconditioner(cond_model, N, device=device)

        case1_iters = []
        case2_iters = []
        case4_iters = []
        case7_iters = []

        for s in range(NUM_SAMPLES):
            f = generate_source_term(X, Y, rng)
            b = assemble_rhs(f, N)

            # Case 1: plain CG
            r1 = conjugate_gradient(A, b, tol=TOL)
            case1_iters.append(r1.iterations)

            # Case 2: warm-start CG
            if ws_model is not None:
                x0 = predict_warmstart(ws_model, f, N, **ws_stats, device=device)
                r2 = conjugate_gradient(A, b, x0=x0, tol=TOL)
                case2_iters.append(r2.iterations)

            # Case 4: IC(0) + PCG
            r4 = preconditioned_cg(A, b, ic0_precond, tol=TOL)
            case4_iters.append(r4.iterations)

            # Case 7: condition-loss + FCG
            if nn_precond is not None:
                r7 = flexible_cg(A, b, nn_precond, tol=TOL, max_iter=1000,
                                 m_max=20)
                case7_iters.append(r7.iterations)

        all_data[f'N{N}'] = {
            'case1': case1_iters,
            'case2': case2_iters if case2_iters else None,
            'case4': case4_iters,
            'case7': case7_iters if case7_iters else None,
        }

        print(f'  Case 1 mean: {np.mean(case1_iters):.2f} (std {np.std(case1_iters):.2f})')
        if case2_iters:
            print(f'  Case 2 mean: {np.mean(case2_iters):.2f} (std {np.std(case2_iters):.2f})')
        print(f'  Case 4 mean: {np.mean(case4_iters):.2f} (std {np.std(case4_iters):.2f})')
        if case7_iters:
            print(f'  Case 7 mean: {np.mean(case7_iters):.2f} (std {np.std(case7_iters):.2f})')

    return all_data


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for the mean of paired differences."""
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def paired_analysis(a: list[int], b: list[int], label: str) -> dict:
    """Run paired Wilcoxon signed-rank test + bootstrap CI on a-b."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diffs = a_arr - b_arr  # positive means a > b (a needs more iters)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    median_diff = float(np.median(diffs))

    # Paired Wilcoxon signed-rank test (two-sided)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs, alternative='two-sided')
    except ValueError:
        # All differences are zero
        wilcoxon_stat, wilcoxon_p = 0.0, 1.0

    # Bootstrap 95% CI for mean paired difference
    ci_lo, ci_hi = bootstrap_ci(diffs)

    # Effect size: matched-pairs rank-biserial correlation
    n = len(diffs)
    if wilcoxon_p < 1.0 and n > 0:
        # r = 1 - (2W / (n*(n+1)/2)) where W is the smaller rank sum
        # Alternatively: r = Z / sqrt(n)
        z_stat = stats.norm.ppf(wilcoxon_p / 2)
        r_effect = abs(z_stat) / np.sqrt(n)
    else:
        r_effect = 0.0

    result = {
        'label': label,
        'n_pairs': n,
        'mean_a': float(np.mean(a_arr)),
        'mean_b': float(np.mean(b_arr)),
        'mean_paired_diff': mean_diff,
        'std_paired_diff': std_diff,
        'median_paired_diff': median_diff,
        'bootstrap_ci_95': [ci_lo, ci_hi],
        'wilcoxon_stat': float(wilcoxon_stat),
        'wilcoxon_p': float(wilcoxon_p),
        'effect_size_r': float(r_effect),
    }

    return result


def holm_bonferroni(results: list[dict]) -> list[dict]:
    """Apply Holm-Bonferroni correction to a list of test results."""
    n = len(results)
    # Sort by p-value
    indexed = sorted(enumerate(results), key=lambda x: x[1]['wilcoxon_p'])

    for rank, (orig_idx, res) in enumerate(indexed):
        adjusted_p = min(res['wilcoxon_p'] * (n - rank), 1.0)
        results[orig_idx]['wilcoxon_p_adjusted'] = adjusted_p
        results[orig_idx]['significant_005'] = adjusted_p < 0.05

    return results


def run_analysis():
    """Main analysis pipeline."""
    print('='*60)
    print('STATISTICAL ANALYSIS — Core Factorial Comparisons')
    print('Paired data across 50 RHS vectors, seed=99')
    print('='*60)

    # Step 1: collect paired data
    data = collect_paired_data()

    # Step 2: run paired tests
    all_results = []

    for N in EVAL_SIZES:
        key = f'N{N}'
        d = data[key]

        if d['case7'] is not None:
            # Case 1 vs Case 7 (CG vs Condition-loss NN)
            r17 = paired_analysis(d['case1'], d['case7'],
                                  f'Case1_vs_Case7_N{N}')
            all_results.append(r17)

            # Case 4 vs Case 7 (IC(0) vs Condition-loss NN)
            r47 = paired_analysis(d['case4'], d['case7'],
                                  f'Case4_vs_Case7_N{N}')
            all_results.append(r47)

        if d['case2'] is not None:
            # Case 1 vs Case 2 (CG vs Warm-start CG)
            r12 = paired_analysis(d['case1'], d['case2'],
                                  f'Case1_vs_Case2_N{N}')
            all_results.append(r12)

    # Step 3: Holm-Bonferroni correction
    all_results = holm_bonferroni(all_results)

    # Step 4: print results
    print('\n' + '='*60)
    print('RESULTS — Paired Wilcoxon Signed-Rank Tests')
    print(f'Holm-Bonferroni corrected across {len(all_results)} tests')
    print('='*60)

    for r in all_results:
        sig = '***' if r['significant_005'] else 'ns'
        print(f'\n{r["label"]}:')
        print(f'  Mean A={r["mean_a"]:.2f}, Mean B={r["mean_b"]:.2f}')
        print(f'  Mean paired diff = {r["mean_paired_diff"]:.2f} '
              f'(95% CI [{r["bootstrap_ci_95"][0]:.2f}, {r["bootstrap_ci_95"][1]:.2f}])')
        print(f'  Wilcoxon W={r["wilcoxon_stat"]:.1f}, '
              f'p={r["wilcoxon_p"]:.2e}, '
              f'p_adj={r["wilcoxon_p_adjusted"]:.2e} {sig}')
        print(f'  Effect size r={r["effect_size_r"]:.3f}')

    # Step 5: Case 6 note
    print('\nCase 6 (MSE-trained) note:')
    print('  All 50 runs capped at 1000 iterations (non-convergent).')
    print('  Zero variance. No meaningful statistical test is possible.')
    print('  Treated as bounded failure, not a comparison.')

    # Step 6: save results
    output = {
        'description': 'Paired statistical tests for core factorial comparisons',
        'method': 'Wilcoxon signed-rank test (paired, two-sided)',
        'correction': 'Holm-Bonferroni',
        'confidence_intervals': 'Bootstrap (10000 resamples, seed=42)',
        'n_samples_per_case': NUM_SAMPLES,
        'random_seed': SEED,
        'n_tests': len(all_results),
        'results': all_results,
        'case6_note': ('All 50 runs capped at 1000 iterations. '
                       'Zero variance. Not statistically testable.'),
    }

    out_path = os.path.join(RESULTS_DIR, 'statistical_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nSaved: {out_path}')

    return output


if __name__ == '__main__':
    run_analysis()
