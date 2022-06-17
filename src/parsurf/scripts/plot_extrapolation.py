#!/usr/bin/env python3

import argparse
import math
import pathlib
from typing import List, Callable, Any

import sinter
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import linregress

from parsurf.tools import score_binomial_line

TITLE_CHANGE = {
    'chao': 'PM Double Ancilla Surface Code (Chao et al. 2020)',
    'honeycomb': 'PM Planar Honeycomb Code (Gidney et al. 2022)',
    'pentagonal_sharp': 'PM Pentagonal Ancilla Surface Code (this paper 2022)',
}
TITLE_ORDER = {
    'chao': 0,
    'pentagonal_sharp': 1,
    'honeycomb': 2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs='+', type=str)
    parser.add_argument("--chunking", required=True, choices=['shot', 'd', 'round'])
    parser.add_argument("--skip_d", default=(), nargs='+', type=int)
    parser.add_argument("--skip_p", default=(), nargs='+', type=float)
    parser.add_argument('--semi_systemic_bayesian', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--save', default=None, type=str)
    args = parser.parse_args()
    if args.save is None and not args.show:
        raise ValueError("Specify --save or --show")

    MARKERS: str = "ov*sp^<>8PhH+xXDd|" * 100
    COLORS: List[str] = list(matplotlib.colors.TABLEAU_COLORS) * 3

    if args.chunking == 'round':
        pieces_func = lambda stat: stat.json_metadata['r']
        per_name = 'round'
    elif args.chunking == 'shot':
        pieces_func = lambda stat: 1
        per_name = 'shot'
    elif args.chunking == 'd':
        pieces_func = lambda stat: math.ceil(stat.json_metadata['r'] / stat.json_metadata['d'])
        per_name = 'd rounds'
    else:
        raise NotImplementedError(f'{args.chunking=}')
    s2p = lambda p, stat: sinter.shot_error_rate_to_piece_error_rate(shot_error_rate=p, pieces=pieces_func(stat))

    samples = sinter.stats_from_csv_files(*args.csv)
    samples = [e for e in samples if e.json_metadata['d'] not in args.skip_d and e.json_metadata['p'] not in args.skip_p]
    marker_key = lambda e: e.json_metadata['p']
    color_key = lambda e: e.json_metadata['p']
    group_func = lambda e: e.json_metadata['c']
    marker_index = {t: k for k, t in enumerate(sorted({marker_key(e) for e in samples}))}
    color_index = {t: k for k, t in enumerate(sorted({color_key(e) for e in samples}))}

    groups = sinter.group_by(samples, key=group_func)
    curve_func = lambda stat: f"{stat.json_metadata['p']}"
    fig, axs = plt.subplots(1, len(groups), gridspec_kw={'wspace': 0.05, 'hspace': 0})
    max_x_lim = 50
    for k, key in enumerate(sorted(groups.keys(), key=TITLE_ORDER.__getitem__)):
        plot_case(
            ax=axs[k],
            samples=groups[key],
            marker_func=lambda e: MARKERS[marker_index[marker_key(e)]],
            color_func=lambda e: COLORS[color_index[color_key(e)]],
            curve_func=curve_func,
            max_likelihood_factor=1000,
            s2p=s2p,
            semi_systemic_bayesian=args.semi_systemic_bayesian,
            max_x_lim=max_x_lim,
        )
        ax: plt.Axes = axs[k]
        ax.set_title(TITLE_CHANGE[key], fontsize=10)
        ax.semilogy()
        ax.set_ylabel(f'Logical Error Rate (per {per_name}) [log scale]')
        ax.set_xlabel('qubits [sqrt scale]')
        x_ticks = range(0, max_x_lim + 10, 10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{e**2}' for e in x_ticks], rotation=70)
        ax.set_yticks([10**d for d in range(-12, 1)])
        if k == 0:
            ax.set_yticks([b*10**d for d in range(-12, 0) for b in range(2, 10)], minor=True)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(1e-12, 1)
        ax.grid(which='major', c='black')
        ax.set_xlim(0, max_x_lim)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.yaxis.label.set_visible(k == 0)
    axs[-1].legend(
        handles=[
            matplotlib.lines.Line2D([0], [0], lw=0, label="Noise")

        ] + [
            matplotlib.lines.Line2D([0], [0], color=COLORS[color_index[p]], lw=1, linestyle='dashed', marker=MARKERS[marker_index[p]], label=str(p))
            for p in sorted({e.json_metadata['p'] for e in samples}, reverse=True)
        ],
        loc="upper left",
        prop={'size': 14},
        bbox_to_anchor=(1, 1),
    )

    fig.set_size_inches(16, 9)
    if args.save is not None:
        pathlib.Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.save, bbox_inches='tight', dpi=256)
        print(f"wrote {args.save}")
    if args.show:
        plt.show()


def plot_case(*,
              ax: plt.Axes,
              samples: List[sinter.TaskStats],
              marker_func: Callable[[sinter.TaskStats], str],
              color_func: Callable[[sinter.TaskStats], str],
              curve_func: Callable[[sinter.TaskStats], str],
              max_likelihood_factor: float,
              s2p: Callable[[float, sinter.TaskStats], float],
              semi_systemic_bayesian: bool,
              max_x_lim: float,
              ) -> None:
    x_func = lambda stat: math.sqrt(stat.json_metadata['q'])

    curves = sinter.group_by(samples, key=curve_func)
    for k, label in enumerate(sorted(curves, key=sinter.better_sorted_str_terms, reverse=True)):
        curve_stats = sorted(curves[label], key=x_func)
        color = color_func(curve_stats[0])
        marker = marker_func(curve_stats[0])

        xs, ys, ys_low, ys_high, ls_xs_fit, ls_ys_fit, sb_xs_fit, sb_shots_fit, sb_errors_fit = categorize_for_fitting(
            curve_stats=curve_stats, s2p=s2p, max_likelihood_factor=max_likelihood_factor)
        ax.scatter(xs, ys, label=label, marker=marker, color=color, linestyle='None')
        for k2 in range(len(xs)):
            ax.plot([xs[k2], xs[k2]], [ys_low[k2], ys_high[k2]], color=color)

        if len(ls_xs_fit) >= 2:
            rep = curve_stats[0]
            log_fit = linregress(ls_xs_fit, [math.log(y) for y in ls_ys_fit])
            if log_fit.slope > -0.02:
                continue

            if semi_systemic_bayesian:
                a, b, c, d = score_binomial_line(
                    min_x_lim=0,
                    max_x_lim=max_x_lim,
                    min_p_lim=1e-12,
                    max_p_lim=1,
                    xs=sb_xs_fit,
                    shots=sb_shots_fit,
                    errors=sb_errors_fit,
                    max_likelihood_factor=max_likelihood_factor,
                    y_distortion=lambda e: s2p(e, rep) if 0 < e < 1 else e,
                )
                ax.plot(a, b, linestyle='dashed', zorder=9, color=color)
                ax.fill(c, d, color=color, alpha=0.25, linewidth=0, zorder=100)
            else:
                ax.plot([0, max_x_lim], [math.exp(log_fit.intercept), math.exp(log_fit.intercept + log_fit.slope * max_x_lim)], linestyle='dashed', zorder=9, color=color)


def categorize_for_fitting(
        *,
        curve_stats: List[sinter.TaskStats],
        s2p: Callable[[float, sinter.TaskStats], float],
        max_likelihood_factor: float) -> Any:
    x_func = lambda stat: math.sqrt(stat.json_metadata['q'])

    xs = []
    ys = []
    ys_low = []
    ys_high = []
    ls_xs_fit = []
    ls_ys_fit = []
    sb_xs_fit = []
    sb_shots_fit = []
    sb_errors_fit = []
    for stat in curve_stats:
        x = x_func(stat)
        y = s2p(stat.errors / stat.shots, stat)

        # Don't show no-observbed-error data points, because their error bars obscure stuff.
        if 0 < stat.errors:
            xs.append(x)
            fit = sinter.fit_binomial(num_shots=stat.shots, num_hits=stat.errors,
                                      max_likelihood_factor=max_likelihood_factor)
            ys.append(s2p(fit.best, stat))
            ys_low.append(s2p(fit.low, stat))
            ys_high.append(s2p(fit.high, stat))

        # Least squares fit can't handle zero probability points (log puts them at negative infinity).
        # Omit points clearly above threshold.
        if 0 < stat.errors < 0.4 * stat.shots:
            ls_xs_fit.append(x)
            ls_ys_fit.append(y)

        # Omit points clearly above threshold.
        if stat.errors < 0.4 * stat.shots:
            sb_xs_fit.append(x)
            if stat.errors > 10:
                sb_shots_fit.append(math.ceil(stat.shots * 10 / stat.errors))
                sb_errors_fit.append(10)
            else:
                sb_shots_fit.append(stat.shots)
                sb_errors_fit.append(stat.errors)

    return xs, ys, ys_low, ys_high, ls_xs_fit, ls_ys_fit, sb_xs_fit, sb_shots_fit, sb_errors_fit


if __name__ == '__main__':
    main()
