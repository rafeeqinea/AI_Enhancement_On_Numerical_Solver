"""Generate all dissertation figures (early chapters + conceptual diagrams)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'report_figures')
os.makedirs(OUTPUT, exist_ok=True)


def fig_taxonomy():
    """Learned-preconditioning taxonomy for Chapter 2."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(6, 5.2, 'Taxonomy of ML-Enhanced Iterative Solvers', ha='center',
            fontsize=13, fontweight='bold')

    top = FancyBboxPatch((3, 4.0), 6, 0.8, boxstyle='round,pad=0.15',
                          facecolor='#E8E8E8', edgecolor='#333', linewidth=1.5)
    ax.add_patch(top)
    ax.text(6, 4.4, 'Hybrid ML + Classical Solver', ha='center', fontsize=10,
            fontweight='bold')

    cats = [
        (1.5, 2.2, 'Warm-Starting', '#5B9BD5',
         'ML predicts initial\nguess x0', 'Before solver'),
        (4.5, 2.2, 'Learned\nPreconditioning', '#ED7D31',
         'ML acts as precond.\ninside solver loop', 'During solver\n(this project)'),
        (7.5, 2.2, 'Learned Solvers', '#A9A9A9',
         'ML replaces the\niteration mechanism', 'Replaces solver'),
        (10.5, 2.2, 'Operator\nLearning', '#A9A9A9',
         'ML maps between\nfunction spaces', 'Bypasses solver'),
    ]

    for cx, cy, title, color, desc, when in cats:
        box = FancyBboxPatch((cx - 1.2, cy - 0.9), 2.4, 1.8,
                              boxstyle='round,pad=0.1', facecolor=color,
                              edgecolor='#333', linewidth=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(cx, cy + 0.35, title, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white')
        ax.text(cx, cy - 0.25, desc, ha='center', va='center', fontsize=7,
                color='white')
        ax.text(cx, cy - 0.75, when, ha='center', va='center', fontsize=6.5,
                color='white', style='italic')
        ax.annotate('', xy=(cx, cy + 0.9), xytext=(cx, 4.0),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

    highlight = patches.FancyBboxPatch((3.15, 1.15), 2.7, 2.1,
                                        boxstyle='round,pad=0.1',
                                        facecolor='none', edgecolor='#ED7D31',
                                        linewidth=2.5, linestyle='--')
    ax.add_patch(highlight)

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_taxonomy.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_factorial_design():
    """Factorial experimental design schematic for Chapter 3."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.text(5.5, 6.7, 'Factorial Experimental Design: 3 Factors, 8 Cases',
            ha='center', fontsize=13, fontweight='bold')

    factors = [
        (2, 5.5, 'Initial Guess', ['Zero start', 'Warm-start (U-Net)'], '#5B9BD5'),
        (5.5, 5.5, 'Preconditioner', ['None / Jacobi', 'IC(0)', 'U-Net'], '#70AD47'),
        (9, 5.5, 'Training Objective', ['n/a', 'MSE', 'Condition loss'], '#ED7D31'),
    ]

    for cx, cy, title, levels, color in factors:
        box = FancyBboxPatch((cx - 1.5, cy - 0.4), 3, 0.8,
                              boxstyle='round,pad=0.1', facecolor=color,
                              edgecolor='#333', linewidth=1.2)
        ax.add_patch(box)
        ax.text(cx, cy, title, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        for i, level in enumerate(levels):
            y = cy - 0.8 - i * 0.45
            ax.text(cx, y, level, ha='center', fontsize=7.5, color='#333',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#F5F5F5',
                             edgecolor='#CCC'))

    cases_data = [
        ('Case 1', 'Zero', 'None', '--', 'CG', '#E8E8E8'),
        ('Case 2', 'WS', 'None', 'MSE', 'CG', '#E8E8E8'),
        ('Case 3', 'Zero', 'Jacobi', '--', 'PCG', '#E8E8E8'),
        ('Case 4', 'Zero', 'IC(0)', '--', 'PCG', '#D4EDDA'),
        ('Case 5', 'WS', 'IC(0)', 'MSE', 'PCG', '#D4EDDA'),
        ('Case 6', 'Zero', 'U-Net', 'MSE', 'FCG', '#F8D7DA'),
        ('Case 7', 'Zero', 'U-Net', 'Cond.', 'FCG', '#FFF3CD'),
        ('Case 8', 'WS', 'U-Net', 'Cond.', 'FCG', '#FFF3CD'),
    ]

    y_start = 2.6
    headers = ['Case', 'x0', 'Precond.', 'Loss', 'Solver']
    col_x = [1.5, 3.2, 4.8, 6.5, 8.0]

    for j, h in enumerate(headers):
        ax.text(col_x[j], y_start + 0.35, h, ha='center', fontsize=8,
                fontweight='bold', color='#333')

    for i, (case, x0, precond, loss, solver, bg) in enumerate(cases_data):
        y = y_start - i * 0.35
        rect = patches.Rectangle((0.5, y - 0.14), 8.5, 0.3, facecolor=bg,
                                   edgecolor='#DDD', linewidth=0.5)
        ax.add_patch(rect)
        vals = [case, x0, precond, loss, solver]
        for j, v in enumerate(vals):
            fw = 'bold' if j == 0 else 'normal'
            ax.text(col_x[j], y, v, ha='center', fontsize=7.5, fontweight=fw)

    ax.text(9.8, 2.4, 'Legend:', fontsize=7.5, fontweight='bold')
    legend_items = [('#D4EDDA', 'Classical'), ('#FFF3CD', 'Condition loss'),
                    ('#F8D7DA', 'MSE (fails)'), ('#E8E8E8', 'Baseline')]
    for i, (c, l) in enumerate(legend_items):
        y = 2.0 - i * 0.35
        rect = patches.Rectangle((9.5, y - 0.1), 0.3, 0.2, facecolor=c,
                                   edgecolor='#999')
        ax.add_patch(rect)
        ax.text(9.95, y, l, fontsize=7, va='center')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_factorial_design.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_training_objectives():
    """Training pipeline comparison for Chapter 4 or 5."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    pipelines = [
        ('MSE Training Pipeline', '#E74C3C',
         [('Random source f', '#E8E8E8'),
          ('Assemble A, b', '#E8E8E8'),
          ('Compute exact correction\nvia direct solver', '#E8E8E8'),
          ('Forward pass: z = UNet(r)', '#5B9BD5'),
          ('MSE Loss: ||z - exact||^2', '#E74C3C'),
          ('Backprop + update weights', '#E8E8E8')],
         'Requires pre-computed\nreference solutions'),
        ('Condition-Loss Training Pipeline', '#ED7D31',
         [('Random probe vectors w', '#E8E8E8'),
          ('Forward pass: z = UNet(w)', '#5B9BD5'),
          ('Apply system matrix: Az', '#E8E8E8'),
          ('Condition Loss:\n||w - Az||^2  ~  ||I - AM||_F^2', '#ED7D31'),
          ('Backprop + update weights', '#E8E8E8')],
         'Self-supervised:\nno reference solutions needed'),
    ]

    for ax, (title, color, steps, note) in zip(axes, pipelines):
        ax.set_xlim(0, 4)
        ax.set_ylim(-0.8, len(steps) * 1.8 + 0.5)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', color=color, pad=10)

        for i, (label, c) in enumerate(steps):
            y = (len(steps) - 1 - i) * 1.8 + 0.5
            box = FancyBboxPatch((0.3, y - 0.4), 3.4, 0.8,
                                  boxstyle='round,pad=0.1', facecolor=c,
                                  edgecolor='#333', linewidth=1)
            ax.add_patch(box)
            tc = 'white' if c in ['#E74C3C', '#ED7D31', '#5B9BD5'] else '#333'
            fw = 'bold' if c != '#E8E8E8' else 'normal'
            ax.text(2, y, label, ha='center', va='center', fontsize=8,
                    color=tc, fontweight=fw)
            if i < len(steps) - 1:
                y_next = (len(steps) - 2 - i) * 1.8 + 0.5
                ax.annotate('', xy=(2, y_next + 0.45), xytext=(2, y - 0.45),
                            arrowprops=dict(arrowstyle='->', color='#333',
                                           lw=1.2))

        ax.text(2, -0.5, note, ha='center', fontsize=8, color=color,
                style='italic',
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor=color + '15', edgecolor=color))

    fig.suptitle('Training Pipeline Comparison: MSE vs Condition Loss',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_training_objectives.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_software_arch():
    """Software architecture diagram for Chapter 4."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(6, 7.2, 'Software Architecture: Modular Experimental Framework',
            ha='center', fontsize=13, fontweight='bold')

    modules = [
        # (cx, cy, w, h, title, items, color)
        (2, 5.5, 2.8, 1.2, 'Data Generation', 'poisson.py\ngenerate.py\ndataset.py', '#5B9BD5'),
        (6, 5.5, 2.8, 1.2, 'Solvers', 'cg.py\npcg.py\nfcg.py\ndirect.py', '#70AD47'),
        (10, 5.5, 2.8, 1.2, 'Preconditioners', 'preconditioners.py\n(Jacobi, IC(0))', '#70AD47'),
        (2, 3.5, 2.8, 1.2, 'Models', 'unet.py\n(dim-generic\n2D/3D)', '#ED7D31'),
        (6, 3.5, 2.8, 1.2, 'Training', 'train.py\nlosses.py\n(MSE, Condition)', '#ED7D31'),
        (10, 3.5, 2.8, 1.2, 'Evaluation', 'evaluate.py\nnn_preconditioner.py\nmetrics.py', '#9B59B6'),
        (6, 1.5, 4, 1.0, 'Experiments', 'run_factorial.py  |  run_condition_loss.py\nrun_curriculum.py  |  run_3d.py', '#E8E8E8'),
    ]

    for cx, cy, w, h, title, items, color in modules:
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.1', facecolor=color,
                              edgecolor='#333', linewidth=1.2)
        ax.add_patch(box)
        tc = 'white' if color != '#E8E8E8' else '#333'
        ax.text(cx, cy + h/2 - 0.22, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=tc)
        ax.text(cx, cy - 0.15, items, ha='center', va='center', fontsize=6.5,
                color=tc, family='monospace')

    # Arrows: data -> solvers, solvers -> preconditioners
    arrows = [
        (3.4, 5.5, 4.6, 5.5),   # data -> solvers
        (7.4, 5.5, 8.6, 5.5),   # solvers -> preconditioners
        (2, 4.9, 2, 4.1),       # data -> models
        (6, 4.9, 6, 4.1),       # solvers -> training
        (10, 4.9, 10, 4.1),     # preconditioners -> evaluation
        (3.4, 3.5, 4.6, 3.5),   # models -> training
        (7.4, 3.5, 8.6, 3.5),   # training -> evaluation
        (6, 2.9, 6, 2.0),       # training -> experiments
    ]

    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1))

    # Output box
    ax.text(6, 0.5, 'results/*.json  +  checkpoints  +  figures',
            ha='center', fontsize=8, color='#555', family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#F0F0F0',
                     edgecolor='#AAA'))
    ax.annotate('', xy=(6, 0.8), xytext=(6, 1.0),
                arrowprops=dict(arrowstyle='->', color='#666', lw=1))

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_software_arch.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_testing_overview():
    """Testing structure diagram for Chapter 4."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    ax.text(5, 5.2, 'Verification Strategy: 161 Tests Across 22 Files',
            ha='center', fontsize=13, fontweight='bold')

    layers = [
        (5, 4.2, 8, 0.7, 'Unit Tests', '#5B9BD5',
         'Poisson assembly shape/symmetry  |  Solver convergence  |  U-Net output dims  |  Loss finite values'),
        (5, 3.2, 8, 0.7, 'Integration Tests', '#70AD47',
         'Model wrapping as preconditioner  |  FCG with NN preconditioner  |  Evaluation pipeline structure'),
        (5, 2.2, 8, 0.7, 'Regression Tests', '#ED7D31',
         'Reproduce Case 1, 3, 4 at N=16 (seed 99)  |  Match committed results.json within tolerance'),
        (5, 1.2, 8, 0.7, 'End-to-End Checks', '#9B59B6',
         'Iterative vs direct solver accuracy  |  Convergent solutions within 1e-6 tolerance'),
    ]

    for cx, cy, w, h, title, color, desc in layers:
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.1', facecolor=color,
                              edgecolor='#333', linewidth=1.2, alpha=0.85)
        ax.add_patch(box)
        ax.text(cx - w/2 + 0.3, cy + 0.05, title, ha='left', va='center',
                fontsize=9, fontweight='bold', color='white')
        ax.text(cx + 0.5, cy - 0.05, desc, ha='center', va='center',
                fontsize=6.5, color='white')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_testing_overview.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_provenance():
    """Results provenance flow for Chapter 5 or appendix."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    ax.text(6, 4.2, 'Experimental Provenance: Scripts to Committed Artefacts',
            ha='center', fontsize=13, fontweight='bold')

    # Scripts (left)
    scripts = [
        (1.5, 3.0, 'run_factorial.py', '#5B9BD5'),
        (1.5, 2.0, 'run_condition_loss.py', '#5B9BD5'),
        (1.5, 1.0, 'run_curriculum.py', '#5B9BD5'),
    ]

    # JSONs (middle)
    jsons = [
        (5.5, 3.0, 'factorial/results.json\n(Cases 1-5, 7, 8)', '#70AD47'),
        (5.5, 2.0, 'condition_results.json\n(Case 7 verification)', '#70AD47'),
        (5.5, 1.0, 'curriculum_results.json\n(N=16 to N=128)', '#70AD47'),
    ]

    # Report (right)
    report = [(9.5, 2.0, 'Chapter 5\nResults Tables\n+ Figures', '#ED7D31')]

    for cx, cy, label, color in scripts + jsons + report:
        w = 2.6 if color == '#5B9BD5' else (3.0 if color == '#70AD47' else 2.4)
        h = 0.65
        box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                              boxstyle='round,pad=0.1', facecolor=color,
                              edgecolor='#333', linewidth=1)
        ax.add_patch(box)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                color='white', fontweight='bold')

    # Arrows: scripts -> jsons
    for i in range(3):
        ax.annotate('', xy=(4.0, scripts[i][1]), xytext=(2.8, scripts[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

    # Arrows: jsons -> report
    for i in range(3):
        ax.annotate('', xy=(8.3, 2.0), xytext=(7.0, jsons[i][1]),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1,
                                   connectionstyle='arc3,rad=0.1'))

    # Seed annotation
    ax.text(3.4, 3.5, 'seed=99, 50 samples', fontsize=7, color='#666',
            style='italic')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_provenance.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('=' * 60)
    print('Generating dissertation figures')
    print('=' * 60)
    fig_taxonomy()
    fig_factorial_design()
    fig_training_objectives()
    fig_software_arch()
    fig_testing_overview()
    fig_provenance()
    print('\nDone. 6 new figures generated.')
