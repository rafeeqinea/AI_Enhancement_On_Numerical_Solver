"""Fix the 3 early figures with spacing/overlay issues."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'report_figures')


def fig_stencil():
    """Improved 5-point stencil with better spacing and equation."""
    fig, ax = plt.subplots(figsize=(6, 6.5))

    # Grid background
    for i in range(5):
        for j in range(5):
            ax.plot(i, j, 'o', color='#DDDDDD', markersize=8, zorder=0)
        ax.axhline(y=i, color='#F0F0F0', linewidth=0.5, zorder=0)
        ax.axvline(x=i, color='#F0F0F0', linewidth=0.5, zorder=0)

    # Stencil connections
    connections = [(2, 2, 1, 2), (2, 2, 3, 2), (2, 2, 2, 1), (2, 2, 2, 3)]
    for x1, y1, x2, y2 in connections:
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)

    # Stencil points
    pts = [
        (2, 2, '4', '#ED7D31', 0.38, '(i, j)'),
        (1, 2, '-1', '#5B9BD5', 0.32, '(i-1, j)'),
        (3, 2, '-1', '#5B9BD5', 0.32, '(i+1, j)'),
        (2, 1, '-1', '#5B9BD5', 0.32, '(i, j-1)'),
        (2, 3, '-1', '#5B9BD5', 0.32, '(i, j+1)'),
    ]

    for x, y, coeff, color, radius, label in pts:
        circle = plt.Circle((x, y), radius, color=color, ec='#222',
                             linewidth=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, coeff, ha='center', va='center', fontsize=15,
                fontweight='bold', color='white', zorder=4)
        # Label below/beside
        if y == 2 and x != 2:
            ax.text(x, y - 0.6, label, ha='center', fontsize=9,
                    color='#444', style='italic')
        elif x == 2 and y != 2:
            if y < 2:
                ax.text(x, y - 0.6, label, ha='center', fontsize=9,
                        color='#444', style='italic')
            else:
                ax.text(x, y + 0.55, label, ha='center', fontsize=9,
                        color='#444', style='italic')
        else:
            ax.text(x + 0.6, y - 0.1, label, ha='left', fontsize=9,
                    color='#444', style='italic')

    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.2, 4.5)
    ax.set_aspect('equal')
    ax.set_title('Five-Point Finite Difference Stencil\nfor the 2D Poisson Equation',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    # Equation as plain text (avoid LaTeX rendering issues)
    eq_text = '-u(i-1,j) - u(i+1,j) - u(i,j-1) - u(i,j+1) + 4u(i,j) = h^2 f(i,j)'
    ax.text(2, -0.9, eq_text, ha='center', fontsize=10, color='#444',
            family='serif',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8',
                     edgecolor='#BBB'))

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_stencil_5point.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_unet():
    """Improved U-Net architecture with better spacing."""
    fig, ax = plt.subplots(figsize=(14, 7))

    y_center = 3.5

    # Encoder blocks
    enc = [
        (1.5, 1.4, 3.2, '16ch', '#5B9BD5'),
        (4.5, 1.2, 2.6, '32ch', '#4A8BC2'),
        (7.5, 1.0, 2.0, '64ch', '#3A7BB0'),
    ]

    # Bottleneck
    bot = (10, 0.7, 1.4, '128ch', '#2D6A9F')

    # Decoder blocks
    dec = [
        (12.5, 1.0, 2.0, '64ch', '#ED7D31'),
        (15.5, 1.2, 2.6, '32ch', '#E06D21'),
        (18.5, 1.4, 3.2, '16ch', '#D35D11'),
    ]

    def draw_block(cx, w, h, label, color):
        rect = FancyBboxPatch((cx - w/2, y_center - h/2), w, h,
                               boxstyle='round,pad=0.08', facecolor=color,
                               edgecolor='#222', linewidth=1.3)
        ax.add_patch(rect)
        ax.text(cx, y_center, label, ha='center', va='center',
                fontsize=10, color='white', fontweight='bold')

    # Draw encoder
    for i, (cx, w, h, label, color) in enumerate(enc):
        draw_block(cx, w, h, label, color)
        if i < 2:
            ax.annotate('', xy=(enc[i+1][0] - enc[i+1][1]/2 - 0.15, y_center),
                        xytext=(cx + w/2 + 0.15, y_center),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
            mid = (cx + w/2 + enc[i+1][0] - enc[i+1][1]/2) / 2
            ax.text(mid, y_center - 0.4, 'pool', ha='center', fontsize=8,
                    color='#666')

    # Arrow: last encoder -> bottleneck
    ax.annotate('', xy=(bot[0] - bot[1]/2 - 0.15, y_center),
                xytext=(enc[-1][0] + enc[-1][1]/2 + 0.15, y_center),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    mid = (enc[-1][0] + enc[-1][1]/2 + bot[0] - bot[1]/2) / 2
    ax.text(mid, y_center - 0.4, 'pool', ha='center', fontsize=8, color='#666')

    # Draw bottleneck
    draw_block(bot[0], bot[1], bot[2], bot[3], bot[4])

    # Arrow: bottleneck -> first decoder
    ax.annotate('', xy=(dec[0][0] - dec[0][1]/2 - 0.15, y_center),
                xytext=(bot[0] + bot[1]/2 + 0.15, y_center),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
    mid = (bot[0] + bot[1]/2 + dec[0][0] - dec[0][1]/2) / 2
    ax.text(mid, y_center - 0.4, 'up', ha='center', fontsize=8, color='#666')

    # Draw decoder
    for i, (cx, w, h, label, color) in enumerate(dec):
        draw_block(cx, w, h, label, color)
        if i < 2:
            ax.annotate('', xy=(dec[i+1][0] - dec[i+1][1]/2 - 0.15, y_center),
                        xytext=(cx + w/2 + 0.15, y_center),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.8))
            mid = (cx + w/2 + dec[i+1][0] - dec[i+1][1]/2) / 2
            ax.text(mid, y_center - 0.4, 'up', ha='center', fontsize=8,
                    color='#666')

    # Skip connections (arcs above)
    for i in range(3):
        enc_cx = enc[i][0]
        dec_cx = dec[2 - i][0]
        enc_h = enc[i][2]
        skip_y = y_center + enc_h/2 + 0.5 + i * 0.3
        ax.annotate('',
                    xy=(dec_cx, skip_y - 0.1),
                    xytext=(enc_cx, skip_y - 0.1),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2,
                                   linestyle='dashed',
                                   connectionstyle=f'arc3,rad=-0.12'))
        mid_x = (enc_cx + dec_cx) / 2
        ax.text(mid_x, skip_y + 0.15, 'skip connection',
                ha='center', fontsize=7, color='#27AE60', fontweight='bold')

    # Labels
    ax.text(4.5, y_center + 2.8, 'Encoder', ha='center', fontsize=13,
            fontweight='bold', color='#5B9BD5')
    ax.text(10, y_center + 1.2, 'Bottleneck', ha='center', fontsize=10,
            color='#2D6A9F', fontweight='bold')
    ax.text(15.5, y_center + 2.8, 'Decoder', ha='center', fontsize=13,
            fontweight='bold', color='#ED7D31')

    # Input/output labels
    ax.text(1.5, y_center - 2.2, 'Input: Residual\n(1, N+2, N+2)',
            ha='center', fontsize=9, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                     edgecolor='#AAA'))
    ax.text(18.5, y_center - 2.2, 'Output: Correction\n(1, N, N)',
            ha='center', fontsize=9, color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0F0F0',
                     edgecolor='#AAA'))

    ax.set_xlim(-0.5, 20)
    ax.set_ylim(0.5, 7)
    ax.set_title('U-Net Preconditioner Architecture (base_features=16, levels=3)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT, 'fig_unet_architecture.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


def fig_solver_pipeline():
    """Improved solver pipeline with better spacing and no overlaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    configs = [
        ('CG\n(No Preconditioner)',
         'Skip (z = r)', '#D5D5D5', 'Identity:\nz = r', None),
        ('PCG\nwith IC(0)',
         'Apply IC(0)\nz = L^-T L^-1 r', '#70AD47',
         'Forward + backward\ntriangular solves', None),
        ('FCG\nwith Condition-Loss U-Net',
         'Apply U-Net\nz = UNet(r/||r||) * ||r||', '#ED7D31',
         'GPU forward pass\n+ unit-norm scaling', 'Orthogonalise\nagainst buffer\n(m_max=20)'),
    ]

    for ax, (title, precond_label, precond_color, detail, extra) in zip(axes, configs):
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.5, 13)
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=12)

        steps = [
            (5, 11.5, 'Compute residual\nr = b - Ax', '#E0E0E0', '#333'),
            (5, 9.2, precond_label, precond_color,
             'white' if precond_color != '#D5D5D5' else '#555'),
            (5, 7.0, 'Update search\ndirection p', '#E0E0E0', '#333'),
            (5, 4.8, 'Update solution\nx = x + ap', '#E0E0E0', '#333'),
            (5, 2.5, 'Converged?\n||r||/||b|| < tol', '#FFF3CD', '#333'),
        ]

        bw, bh = 4.0, 1.4

        for cx, cy, label, bg, tc in steps:
            box = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                                  boxstyle='round,pad=0.15', facecolor=bg,
                                  edgecolor='#333', linewidth=1.2)
            ax.add_patch(box)
            fw = 'bold' if bg not in ['#E0E0E0', '#FFF3CD'] else 'normal'
            ax.text(cx, cy, label, ha='center', va='center', fontsize=9,
                    color=tc, fontweight=fw)

        # Arrows between steps
        for i in range(4):
            y1 = steps[i][1] - bh/2
            y2 = steps[i+1][1] + bh/2
            ax.annotate('', xy=(5, y2 + 0.05), xytext=(5, y1 - 0.05),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.3))

        # Flow labels
        ax.text(5.3, 10.3, 'r', fontsize=10, style='italic', color='#555',
                fontweight='bold')
        ax.text(5.3, 8.0, 'z', fontsize=10, style='italic', color='#555',
                fontweight='bold')

        # Detail annotation under preconditioner
        ax.text(5, 9.2 - bh/2 - 0.35, detail, ha='center', fontsize=7.5,
                color='#555', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#FAFAFA',
                         edgecolor='#DDD'))

        # Loop-back arrow
        ax.text(7.8, 2.5, 'No', fontsize=9, color='#CC0000', fontweight='bold')
        ax.annotate('', xy=(5 + bw/2 + 0.8, 11.5),
                    xytext=(5 + bw/2 + 0.8, 2.5),
                    arrowprops=dict(arrowstyle='->', color='#CC0000', lw=1.3,
                                   connectionstyle='arc3,rad=0.25'))

        # Yes output
        ax.annotate('', xy=(5, 1.3), xytext=(5, 2.5 - bh/2 - 0.05),
                    arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=1.3))
        ax.text(5, 0.8, 'Yes: Solution x', fontsize=10, color='#2E8B57',
                fontweight='bold', ha='center')

        # FCG extra step
        if extra:
            ax.text(1.3, 7.0, extra, fontsize=7.5, color='#8B4513',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF5EB',
                             edgecolor='#DEB887', linewidth=1.2))
            ax.annotate('', xy=(5 - bw/2 - 0.05, 7.0),
                        xytext=(3.0, 7.0),
                        arrowprops=dict(arrowstyle='->', color='#DEB887', lw=1.2))

    fig.suptitle('Solver Pipeline: How the Preconditioner Integrates into the Iteration Loop',
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT, 'fig_solver_pipeline.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


if __name__ == '__main__':
    print('Fixing 3 early figures...')
    fig_stencil()
    fig_unet()
    fig_solver_pipeline()
    print('Done.')
