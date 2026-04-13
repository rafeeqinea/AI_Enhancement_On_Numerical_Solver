"""Derive per-iteration timing and overhead ratios from committed factorial results."""
import json, os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def run():
    # Load factorial results
    with open(os.path.join(RESULTS_DIR, 'factorial', 'results.json')) as f:
        factorial = json.load(f)
    # Load AMG results
    with open(os.path.join(RESULTS_DIR, 'baseline', 'amg_results.json')) as f:
        amg = json.load(f)

    breakdown = {}
    for N in [16, 32, 64]:
        key = f'N{N}'
        f_data = factorial[key]
        a_data = amg[key]

        cases = {}
        for case_name, label in [('Case 1', 'CG'), ('Case 4', 'IC(0)+PCG'),
                                  ('Case 7', 'Cond+FCG')]:
            if case_name in f_data:
                d = f_data[case_name]
                per_iter = d['mean_time'] / max(d['mean_iters'], 1)
                cases[case_name] = {
                    'label': label,
                    'mean_iters': d['mean_iters'],
                    'total_time_s': d['mean_time'],
                    'per_iteration_ms': per_iter * 1000,
                    'overhead_vs_cg': d['mean_time'] / max(f_data['Case 1']['mean_time'], 1e-12),
                }

        # AMG
        cases['AMG'] = {
            'label': 'AMG (PyAMG)',
            'mean_iters': a_data['mean_iters'],
            'total_time_s': a_data['mean_time'],  # already in seconds
            'per_iteration_ms': a_data['mean_time'] / max(a_data['mean_iters'], 1) * 1000,
            'overhead_vs_cg': a_data['mean_time'] / max(f_data['Case 1']['mean_time'], 1e-12),
        }

        breakdown[key] = cases

    # Save
    os.makedirs(os.path.join(RESULTS_DIR, 'analysis'), exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'analysis', 'wallclock_breakdown.json')
    with open(out_path, 'w') as f:
        json.dump(breakdown, f, indent=2)

    # Print summary
    print('Wall-Clock Breakdown (derived from committed results)')
    print('=' * 70)
    for N in [16, 32, 64]:
        key = f'N{N}'
        print(f'\nN={N}:')
        print(f'  {"Method":<15} {"Iters":>6} {"Total(ms)":>10} {"Per-iter(ms)":>12} {"vs CG":>8}')
        print(f'  {"-"*55}')
        for case_name in ['Case 1', 'Case 4', 'Case 7', 'AMG']:
            if case_name in breakdown[key]:
                d = breakdown[key][case_name]
                print(f'  {d["label"]:<15} {d["mean_iters"]:>6.1f} '
                      f'{d["total_time_s"]*1000:>10.2f} '
                      f'{d["per_iteration_ms"]:>12.3f} '
                      f'{d["overhead_vs_cg"]:>8.1f}x')

    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    run()
