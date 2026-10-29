"""Iso-lines of the analytical mean tracer velocity of Model M13 over R and S (Figure 4)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

R_MIN, R_MAX = 1e-3, 10.0
ISO_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
LABELED_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
GUIDE_VALUES = np.logspace(-3, 1, 5)
GUIDE_COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30']


def velocity_ratio(R, S):
    """v_mean / v_conv for Model M13 without axial dispersion (Eq. 20)."""
    return 1.0 / (1.0 + R / (R + S) ** 2)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False})

    # extended grid, so that the iso-lines are not clipped at the plot boundary
    grid = np.logspace(np.log10(R_MIN) - 0.5, np.log10(R_MAX) + 0.6, 700)
    RR, SS = np.meshgrid(grid, grid)
    norm = plt.Normalize(0, 1)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    contours = ax.contour(RR, SS, velocity_ratio(RR, SS), levels=ISO_LEVELS,
                          cmap=plt.cm.plasma, norm=norm, linewidths=1.6)
    # labels where the iso-lines cross R = S, i.e. at R = L / (4 (1 - L))
    label_positions = [(L / (4 * (1 - L)),) * 2 for L in LABELED_LEVELS]
    labels = ax.clabel(contours, fmt={L: f'{L:.1f}' for L in LABELED_LEVELS},
                       manual=label_positions, fontsize=9, inline=True)
    for label in labels:
        label.set_bbox(dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))
        label.set_rotation(45)

    for value, color in zip(GUIDE_VALUES, GUIDE_COLORS):
        ax.axhline(value, color=color, lw=0.9, ls='--', alpha=0.7)
        ax.axvline(value, color=color, lw=0.9, ls=':', alpha=0.7)
    diagonal = np.logspace(np.log10(R_MIN), np.log10(R_MAX), 400)
    ax.plot(diagonal, diagonal, 'k-', lw=1.3, alpha=0.5, label=r'$R = S$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(R_MIN, R_MAX)
    ax.set_ylim(R_MIN, R_MAX)
    ax.set_xlabel(r'$R = e_{21}/e_{12}$  (back-exchange)')
    ax.set_ylabel(r'$S = e_{23}/e_{12}$  (storage / loss)')
    ax.legend(fontsize=9, loc='lower right')
    colorbar = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    fig.colorbar(colorbar, ax=ax, label=r'$v_{\mathrm{mean}}\,/\,v_{\mathrm{conv}}$', pad=0.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'vmean_isolines.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    main(parser.parse_args().output_dir)
