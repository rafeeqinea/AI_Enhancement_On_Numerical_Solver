"""Build test inventory with categories from the test suite."""
import json, os, re, glob

TESTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

CATEGORY_MAP = {
    'test_cg.py': 'solver',
    'test_pcg.py': 'solver',
    'test_fcg.py': 'solver',
    'test_direct.py': 'solver',
    'test_poisson.py': 'data/operator',
    'test_poisson_3d.py': '3D',
    'test_dataset.py': 'data/operator',
    'test_generate.py': 'data/operator',
    'test_preconditioners.py': 'preconditioner',
    'test_nn_preconditioner.py': 'preconditioner',
    'test_unet.py': 'model',
    'test_losses.py': 'training/loss',
    'test_condition_loss_pipeline.py': 'training/loss',
    'test_train.py': 'training',
    'test_evaluate.py': 'evaluation',
    'test_evaluate_precond.py': 'evaluation',
    'test_metrics.py': 'evaluation',
    'test_visualize.py': 'evaluation',
    'test_factorial_cases.py': 'regression/structure',
    'test_regression.py': 'regression/reproducibility',
    'test_variable_poisson.py': 'variable-coefficient',
}


def count_tests(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    return len(re.findall(r'def test_', content))


def run():
    inventory = {'total_tests': 0, 'total_files': 0, 'categories': {}, 'files': []}

    for test_file in sorted(glob.glob(os.path.join(TESTS_DIR, 'test_*.py'))):
        fname = os.path.basename(test_file)
        n_tests = count_tests(test_file)
        category = CATEGORY_MAP.get(fname, 'other')

        inventory['files'].append({
            'file': fname,
            'category': category,
            'test_count': n_tests,
        })
        inventory['total_tests'] += n_tests
        inventory['total_files'] += 1
        inventory['categories'][category] = (
            inventory['categories'].get(category, 0) + n_tests
        )

    # Save
    os.makedirs(os.path.join(RESULTS_DIR, 'testing'), exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'testing', 'test_inventory.json')
    with open(out_path, 'w') as f:
        json.dump(inventory, f, indent=2)

    # Print
    print(f'Test Inventory: {inventory["total_tests"]} tests in {inventory["total_files"]} files')
    print('=' * 50)
    for cat, count in sorted(inventory['categories'].items(), key=lambda x: -x[1]):
        print(f'  {cat:<30} {count:>3} tests')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    run()
