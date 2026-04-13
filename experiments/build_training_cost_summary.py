"""Aggregate training costs from committed config/log files."""
import json, os, glob

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def run():
    summary = {'stages': [], 'total_training_time_s': 0, 'total_training_time_h': 0}

    # Condition-loss training (core)
    for N in [16, 32, 64]:
        log_path = os.path.join(RESULTS_DIR, 'nn_precond',
                                f'condition_checkpoints_N{N}', 'training_log.json')
        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
            epochs = len(log.get('train_loss', []))
            summary['stages'].append({
                'name': f'Condition-loss N={N}',
                'type': 'condition_loss_core',
                'epochs': epochs,
                'training_time_s': None,  # not in training_log
            })

    # MSE training
    for N in [16, 32, 64]:
        log_path = os.path.join(RESULTS_DIR, 'nn_precond',
                                f'mse_checkpoints_N{N}', 'training_log.json')
        if os.path.exists(log_path):
            summary['stages'].append({
                'name': f'MSE N={N}',
                'type': 'mse_core',
                'training_time_s': None,
            })

    # Curriculum training (has config.json with times)
    for N in [16, 32, 64, 128]:
        cfg_path = os.path.join(RESULTS_DIR, 'curriculum', '2d', f'N{N}', 'config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            t = cfg.get('training_time_s', 0)
            summary['stages'].append({
                'name': f'Curriculum 2D N={N}',
                'type': 'curriculum_2d',
                'epochs': cfg.get('epochs'),
                'is_scratch': cfg.get('is_scratch', False),
                'training_time_s': t,
            })
            summary['total_training_time_s'] += t

    # 3D training
    cfg_path = os.path.join(RESULTS_DIR, 'checkpoints', '3d',
                            'condition_3d_N32', 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        t = cfg.get('training_time_s', 0)
        summary['stages'].append({
            'name': '3D Condition-loss N=32',
            'type': '3d',
            'epochs': cfg.get('epochs'),
            'training_time_s': t,
        })
        summary['total_training_time_s'] += t

    # Warm-start training
    ws_log = os.path.join(RESULTS_DIR, 'warmstart', 'unet_checkpoints', 'training_log.json')
    if os.path.exists(ws_log):
        summary['stages'].append({
            'name': 'Warm-start U-Net',
            'type': 'warmstart',
            'training_time_s': None,
        })

    summary['total_training_time_h'] = summary['total_training_time_s'] / 3600
    summary['gpu_model'] = 'NVIDIA RTX 4060 Laptop GPU'
    summary['estimated_tdp_w'] = 115
    summary['estimated_energy_kwh'] = (
        summary['total_training_time_h'] * 115 / 1000
    )

    # Save
    os.makedirs(os.path.join(RESULTS_DIR, 'analysis'), exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'analysis', 'training_cost_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print
    print('Training Cost Summary')
    print('=' * 60)
    for stage in summary['stages']:
        t = stage.get('training_time_s')
        t_str = f'{t/60:.1f} min' if t else 'unknown'
        print(f'  {stage["name"]:<30} {t_str}')
    print(f'\nTotal committed training time: {summary["total_training_time_s"]/3600:.2f} hours')
    print(f'Estimated energy: {summary["estimated_energy_kwh"]:.3f} kWh')
    print(f'(assuming {summary["estimated_tdp_w"]}W TDP)')
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    run()
