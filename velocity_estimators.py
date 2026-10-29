"""Simulation study: velocity estimators for Model M13 (Table 1, Figures 5-7).

Simulates tracer pulses with the MCT model in CADET, determines reference times
with four estimators at three observation positions, and fits the velocity.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, RANSACRegressor

from CADETProcess.processModel import (
    ComponentSystem, FlowSheet, Inlet, LumpedRateModelWithoutPores,
    MassActionLaw, MCT, Outlet, Process,
)
from CADETProcess.simulator import Cadet

V_CONV = 5e-5                  # m/s, convective velocity (3 mm/min)
K_DECAY = 0.00056729782996132  # 1/s, decay constant of 11C
LENGTH = 0.2                   # m, length of the MCT segment
POSITIONS = [0.03, 0.06, 0.09]  # m, observation positions
CYCLE_TIME = 200 * 60          # s

SCENARIOS = [
    dict(
        filename='waterfall_exchange.pdf',
        title='Back-exchange',
        low=dict(label='Low back-exchange\n' r'$e_{12}{=}e_{21}{=}1\times10^{-3}$',
                 disp=1e-12, e12=1e-3, e21=1e-3, e23=1e-2),
        high=dict(label='High back-exchange\n' r'$e_{12}{=}e_{21}{=}1\times10^{-2}$',
                  disp=1e-12, e12=1e-2, e21=1e-2, e23=1e-6),
    ),
    dict(
        filename='waterfall_dispersion.pdf',
        title='Dispersion',
        low=dict(label='Low dispersion\n' r'$D{=}10^{-7}\ \mathrm{m^2/s}$',
                 disp=1e-7, e12=0.0, e21=0.0, e23=0.0),
        high=dict(label='High dispersion\n' r'$D{=}10^{-6}\ \mathrm{m^2/s}$',
                  disp=1e-6, e12=0.0, e21=0.0, e23=0.0),
    ),
    dict(
        filename='waterfall_both.pdf',
        title='Back-exchange + Dispersion',
        low=dict(label='Low back-exchange + dispersion\n' r'$e{=}1\times10^{-4}$, $D{=}10^{-7}$',
                 disp=1e-7, e12=1e-4, e21=1e-4, e23=1e-3),
        high=dict(label='High back-exchange + dispersion\n' r'$e{=}1\times10^{-2}$, $D{=}10^{-6}$',
                  disp=1e-6, e12=1e-2, e21=1e-2, e23=1e-3),
    ),
]

ESTIMATORS = {
    'centroid': dict(color='k', ls='-', marker='D', label=r'$t_{\mathrm{mean}}$ (numerical)'),
    'integral': dict(color='#0072BD', ls='--', marker='o', label=r'$t_{\mathrm{half\text{-}integral}}$'),
    'half_max': dict(color='#D95319', ls='-.', marker='s', label=r'$t_{\mathrm{half\text{-}max}}$'),
    'intercept': dict(color='#EDB120', ls=':', marker='^', label=r'$t_{\mathrm{intercept}}$'),
}
ZORDERS = {'centroid': 3, 'integral': 5, 'half_max': 5, 'intercept': 5}

PULSE_SCALE = 37.0  # mm, height of a normalized pulse in the waterfall plot
PLOT_POSITIONS = [p * 1000 for p in reversed(POSITIONS)]
POSITION_LABELS = [f'ROI {i + 1}\n$x = {p * 1000:.0f}$ mm' for i, p in enumerate(POSITIONS)]
YLIM = (0.6 * PULSE_SCALE, max(PLOT_POSITIONS) + 1.5 * PULSE_SCALE)


def setup_process(dispersion, e12, e21, e23):
    """Inlet pulse shaped by a short LRM, followed by MCT model M13."""
    component_system = ComponentSystem(1)
    flow_sheet = FlowSheet(component_system)

    inlet = Inlet(component_system, name='inlet')
    leaf = LumpedRateModelWithoutPores(component_system, name='leaf')
    mct = MCT(component_system, nchannel=3, name='mct')
    outlet = Outlet(component_system, name='outlet')

    decay = MassActionLaw(component_system)
    decay.add_reaction(indices=[0], coefficients=[-1], k_fwd=K_DECAY, k_bwd=0)
    mct.bulk_reaction_model = decay
    leaf.particle_reaction_model = decay

    for unit in [inlet, leaf, mct, outlet]:
        flow_sheet.add_unit(unit)
    flow_sheet.add_connection(inlet, leaf)
    flow_sheet.add_connection(leaf, mct, destination_port='channel_0')
    flow_sheet.add_connection(mct, outlet, origin_port='channel_0')

    inlet.flow_rate = V_CONV

    leaf.total_porosity = 0.4
    leaf.length = 0.01
    leaf.diameter = np.sqrt(1 / np.pi) * 8
    leaf.axial_dispersion = 1e-8
    leaf.discretization.ncol = 100

    mct.length = LENGTH
    mct.channel_cross_section_areas = [1, 1, 1]
    mct.axial_dispersion = [dispersion, 0.0, 0.0]  # only the convective channel disperses
    mct.exchange_matrix = np.array([
        [[0.0], [e12], [0.0]],
        [[e21], [0.0], [e23]],
        [[0.0], [0.0], [0.0]],
    ])
    mct.discretization.ncol = 200
    mct.solution_recorder.write_solution_bulk = 1

    process = Process(flow_sheet, 'tracer_transport')
    process.cycle_time = CYCLE_TIME
    process.add_event('start', 'flow_sheet.inlet.c', [100], 0)
    process.add_event('stop', 'flow_sheet.inlet.c', [0], 60)
    return process


def half_integral_time(pulse, time):
    """Time at which the cumulative integral reaches 50 % of its total."""
    cumulative = np.concatenate(([0], np.cumsum((pulse[:-1] + pulse[1:]) / 2 * np.diff(time))))
    cumulative /= cumulative[-1]
    i = np.searchsorted(cumulative, 0.5)
    if i == 0:
        return time[0]
    return time[i - 1] + (0.5 - cumulative[i - 1]) * (time[i] - time[i - 1]) / (cumulative[i] - cumulative[i - 1])


def half_max_time(pulse, time):
    """Time of the first crossing of half the pulse maximum."""
    half_max = 0.5 * np.max(pulse)
    i = np.where(pulse >= half_max)[0][0]
    if i == 0:
        return time[0]
    return time[i - 1] + (half_max - pulse[i - 1]) * (time[i] - time[i - 1]) / (pulse[i] - pulse[i - 1])


def intercept_time(pulse, time, threshold=0.1):
    """Intersection of a RANSAC line fitted to the rising edge with zero activity."""
    start = np.where(pulse > threshold * np.max(pulse))[0][0]
    rising = np.arange(start, np.argmax(pulse) + 1)
    model = RANSACRegressor(estimator=LinearRegression(), residual_threshold=0.01, random_state=0)
    model.fit(time[rising].reshape(-1, 1), pulse[rising])
    return -model.estimator_.intercept_ / model.estimator_.coef_[0]


def centroid_time(pulse, time, rel_threshold=1e-8):
    """Temporal centroid (mean arrival time) of the pulse."""
    mask = pulse > rel_threshold * np.max(pulse)
    return np.trapezoid(pulse[mask] * time[mask], time[mask]) / np.trapezoid(pulse[mask], time[mask])


def run_scenario(simulator, disp, e12, e21, e23):
    """Simulate a scenario and return decay-corrected mobile-tracer pulses and reference times."""
    results = simulator.simulate(setup_process(disp, e12, e21, e23))
    time = results.time_complete
    mobile = results.solution.mct.bulk.solution[:, :, 0, 0] * np.exp(K_DECAY * time)[:, None]

    ncol = mobile.shape[1]
    columns = [min(int(round(x / (LENGTH / ncol))), ncol - 1) for x in POSITIONS]
    raw = [mobile[:, c] for c in columns]
    normalized = [p / np.max(p) for p in raw]
    amplitudes = [np.max(p) / np.max(raw[0]) for p in raw]

    times = {
        'centroid': [centroid_time(p, time) for p in raw],
        'integral': [half_integral_time(p, time) for p in normalized],
        'half_max': [half_max_time(p, time) for p in normalized],
        'intercept': [intercept_time(p, time) for p in normalized],
    }
    return time, normalized, amplitudes, times


def fit_velocity(reference_times):
    """Slope of x = v t + b and its standard error, both in mm/min."""
    fit = linregress(reference_times, POSITIONS)
    return fit.slope * 60000, fit.stderr * 60000


def format_se(se):
    return '$<$0.01' if se < 0.01 else f'= {se:.2f}'


def mean_velocity_analytical(e12, e21, e23, dispersion=0.0, v=V_CONV):
    """Average tracer velocity of Model M13 (Eqs. 18 and 27), in mm/min."""
    if e12 == 0:
        return v * 60000
    k = e12 * e23 / (e21 + e23)
    q = e12 * e21 / (e21 + e23) ** 2
    return np.sqrt(v ** 2 + 4 * dispersion * k) / (1 + q) * 60000


def draw_waterfall(ax, time, pulses, amplitudes, times, left):
    t_min = time / 60
    for position, pulse in zip(PLOT_POSITIONS, pulses):
        y = pulse * PULSE_SCALE + position
        ax.plot(t_min, y, color='#77AC30', lw=1.4, alpha=0.9)
        ax.fill_between(t_min, position, y, alpha=0.18, color='#77AC30')
        ax.axhline(position, color='gray', lw=0.6, alpha=0.35, zorder=0)
        ax.axhline(position + PULSE_SCALE, color='gray', lw=0.5, ls=':', alpha=0.45, zorder=0)

    handles = []
    for name, style in ESTIMATORS.items():
        v, se = fit_velocity(times[name])
        t_ref = [t / 60 for t in times[name]]
        ax.plot(t_ref, PLOT_POSITIONS, color=style['color'], ls=style['ls'], lw=2.0,
                zorder=ZORDERS[name], marker=style['marker'], markersize=8,
                markeredgecolor='k', markeredgewidth=0.6, label=style['label'])
        for position, pulse, t in zip(PLOT_POSITIONS, pulses, t_ref):
            top = np.interp(t, t_min, pulse) * PULSE_SCALE + position
            ax.vlines(t, position, top, color=style['color'], lw=1.4, alpha=0.85, zorder=ZORDERS[name])
        handles.append(Line2D([0], [0], color=style['color'], ls=style['ls'], lw=1.5,
                              marker=style['marker'], markersize=6, markeredgecolor='k',
                              markeredgewidth=0.5, label=f'{v:.2f} (SE {format_se(se)})'))

    ax.legend(handles=handles, loc='upper right', fontsize=12, framealpha=0.92,
              title='Estimated $v$ [mm/min]', title_fontsize=9)
    ax.set_xlabel('Time [min]', fontsize=13)
    ax.set_xlim(0, CYCLE_TIME / 60)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(YLIM)
    ax.set_yticks(PLOT_POSITIONS)
    if left:
        ax.set_ylabel('Position [mm]', fontsize=13)
        ax.set_yticklabels(POSITION_LABELS, fontsize=10)
    else:
        ax.set_yticklabels([])

    trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for position, amplitude in zip(PLOT_POSITIONS, amplitudes):
        ax.text(0.01, position, '0', transform=trans, fontsize=8, color='gray', va='center')
        ax.text(0.01, position + PULSE_SCALE, f'{amplitude:.2f}', transform=trans,
                fontsize=8, color='gray', va='center')


def draw_reference_times(ax, times_low, times_high, title):
    for times, ls, filled, zorder in [(times_low, '--', False, 4), (times_high, '-', True, 5)]:
        for name, style in ESTIMATORS.items():
            color = style['color']
            ax.scatter([t / 60 for t in times[name]], [p * 1000 for p in POSITIONS],
                       marker=style['marker'], s=60, zorder=zorder, edgecolors=color,
                       facecolors=color if filled else 'white', linewidths=0.9)
            fit = linregress(times[name], POSITIONS)
            t_fit = np.array([min(times[name]) * 0.85, max(times[name]) * 1.15])
            ax.plot(t_fit / 60, (fit.slope * t_fit + fit.intercept) * 1000, color=color, ls=ls, lw=1.8, zorder=3)

    ax.legend(handles=[Line2D([0], [0], color='gray', ls='--', lw=2, label='Low'),
                       Line2D([0], [0], color='gray', ls='-', lw=2, label='High')],
              fontsize=9, loc='upper left', title=title, title_fontsize=9)
    ax.set_xlabel('Reference time [min]', fontsize=13)
    ax.set_yticklabels([])
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title('$t_\\mathrm{ref}$ vs. position', fontsize=11, pad=8)
    ax.set_ylim(YLIM)
    ax.set_yticks(PLOT_POSITIONS)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    simulator = Cadet()
    simulator.time_resolution = 1

    rows = []
    for scenario in SCENARIOS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True,
                                 gridspec_kw={'wspace': 0.05, 'width_ratios': [2, 2, 1.4]})
        times = {}
        for i, case in enumerate(['low', 'high']):
            params = scenario[case]
            time, pulses, amplitudes, times[case] = run_scenario(
                simulator, params['disp'], params['e12'], params['e21'], params['e23'])
            draw_waterfall(axes[i], time, pulses, amplitudes, times[case], left=(i == 0))
            axes[i].set_title(params['label'], fontsize=11, pad=8)

            row = [scenario['title'], case,
                   mean_velocity_analytical(params['e12'], params['e21'], params['e23']),
                   mean_velocity_analytical(params['e12'], params['e21'], params['e23'], params['disp'])]
            for name in ESTIMATORS:
                row += list(fit_velocity(times[case][name]))
            rows.append(row)

        draw_reference_times(axes[2], times['low'], times['high'], scenario['title'])
        axes[0].set_yticklabels(POSITION_LABELS, fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=13,
                   bbox_to_anchor=(0.5, -0.10), framealpha=0.9)
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(output_dir / scenario['filename'], bbox_inches='tight')
        plt.close(fig)

    header = ('scenario,case,analytical_dispersion_free,analytical_with_dispersion,'
              + ','.join(f'{name},{name}_se' for name in ESTIMATORS))
    lines = [header] + [','.join([r[0], r[1]] + [f'{x:.6f}' for x in r[2:]]) for r in rows]
    (output_dir / 'table1_velocities_mm_per_min.csv').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    main(parser.parse_args().output_dir)
