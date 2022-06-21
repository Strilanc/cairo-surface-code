#!/usr/bin/env python3

import argparse
import math
import pathlib
from typing import List, Optional, Tuple, Callable

import matplotlib.colors
import numpy as np
import sinter
from matplotlib import pyplot as plt
from scipy.stats import linregress

from parsurf.scripts.plot_extrapolation import categorize_for_fitting, TITLE_ORDER, TITLE_CHANGE
from parsurf.tools import score_binomial_line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs='+', type=str)
    parser.add_argument("--chunking", required=True, choices=['shot', 'd', 'round'])
    parser.add_argument("--skip_d", default=(), nargs='+', type=int)
    parser.add_argument("--skip_p", default=(), nargs='+', type=float)
    parser.add_argument("--skip_b", default=(), nargs='+', type=str)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--semi_systemic_bayesian', action='store_true')
    parser.add_argument('--save', default=None, type=str)

    args = parser.parse_args()
    if args.save is None and not args.show:
        raise ValueError("Specify --save or --show")
    MARKERS: str = "ov*sp^<>8PhH+xXDd|" * 100
    COLORS: List[str] = list(matplotlib.colors.TABLEAU_COLORS) * 3

    samples = sinter.stats_from_csv_files(*args.csv)
    samples = [e for e in samples
               if e.json_metadata['d'] not in args.skip_d
               and e.json_metadata['p'] not in args.skip_p
               and e.json_metadata['b'] not in args.skip_b]

    if args.chunking == 'round':
        pieces_func = lambda stat: stat.json_metadata['r']
        per_name = 'rounds'
    elif args.chunking == 'shot':
        pieces_func = lambda stat: 1
        per_name = 'shots'
    elif args.chunking == 'd':
        pieces_func = lambda stat: math.ceil(stat.json_metadata['r'] / stat.json_metadata['d'])
        per_name = 'd rounds'
    else:
        raise NotImplementedError(f'{args.chunking=}')
    s2p = lambda p, stat: sinter.shot_error_rate_to_piece_error_rate(shot_error_rate=p, pieces=pieces_func(stat))

    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots(1, 1)

    def fit_group_func(stat: sinter.TaskStats) -> Tuple[str, float]:
        m = stat.json_metadata
        return f"{m['c']}", m['p']

    fit_groups = sinter.group_by(samples, key=fit_group_func)
    data_points: List[Tuple[str, float, sinter.Fit]] = []
    for (label, p), fit_group in fit_groups.items():
        q = fit_teraquop_intercept(fit_group, s2p, semi_systemic_bayesian=args.semi_systemic_bayesian)
        if q is None:
            continue
        data_points.append((label, p, q))

    curves = sinter.group_by(data_points, key=lambda e: e[0])
    for k, label in enumerate(sorted(curves, key=TITLE_ORDER.__getitem__)):
        xs = []
        ys = []
        xs_range = []
        ys_low = []
        ys_high = []
        for _, p, fit in sorted(curves[label], key=lambda e: e[1]):
            xs.append(p)
            ys.append(fit.best)
            xs_range.append(p)
            ys_low.append(fit.low)
            ys_high.append(fit.high)

        ax.plot(xs, ys, label=TITLE_CHANGE[label], marker=MARKERS[k], color=COLORS[k])
        ax.fill_between(xs_range, ys_low, ys_high, alpha=0.25, color=COLORS[k])
    ax.loglog()
    ax.set_ylim(1e2, 1e5)
    ax.set_xlim(1e-4, 1e-2)
    ax.grid(which='major', c='black')
    ax.grid(which='minor')
    ax.set_ylabel(f'Qubits needed for 1 error per trillion {per_name}')
    ax.set_xlabel('Physical error rate')
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    ax.legend(prop={'size': 14})

    fig.set_size_inches(16, 9)
    if args.save is not None:
        pathlib.Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.save, bbox_inches='tight', dpi=256)
        print(f"wrote {args.save}")
    if args.show:
        plt.show()


def fit_teraquop_intercept(stats: List[sinter.TaskStats],
                           s2p: Callable[[float, sinter.TaskStats], float],
                           semi_systemic_bayesian: bool = False) -> Optional[sinter.Fit]:
    max_likelihood_factor = 1000

    _, _, _, _, ls_xs_fit, ls_ys_fit, sb_xs_fit, sb_shots_fit, sb_errors_fit = categorize_for_fitting(
        curve_stats=stats, s2p=s2p, max_likelihood_factor=max_likelihood_factor)

    if len(ls_xs_fit) < 2:
        return None
    log_fit = linregress(ls_xs_fit, [math.log(y) for y in ls_ys_fit])
    if log_fit.slope > -0.02:
        return None

    if semi_systemic_bayesian:
        rep = stats[0]
        a, b, c, d = score_binomial_line(
            min_x_lim=0,
            max_x_lim=1000,
            min_p_lim=1e-12,
            max_p_lim=1,
            xs=sb_xs_fit,
            shots=sb_shots_fit,
            errors=sb_errors_fit,
            max_likelihood_factor=max_likelihood_factor,
            y_distortion=lambda e: s2p(e, rep) if 0 < e < 1 else e,
        )
        cxs = []
        for k in range(1, len(c)):
            if (d[k - 1] < 1e-12) != (d[k] < 1e-12) and k != len(c) // 2:
                vy0 = np.log(d[k - 1])
                vy1 = np.log(d[k])
                vx0 = c[k - 1]
                vx1 = c[k]
                cxs.append(vx0 + ((vx0 - vx1) / (vy0 - vy1)) * (math.log(1e-12) - vy0))
        if len(cxs) == 2:
            low, high = sorted(cxs)
        elif len(cxs) == 1:
            low, = cxs
            high = 10000
        else:
            raise NotImplementedError()
        x0, x1 = a
        y0, y1 = b
        y0 = np.log(y0)
        y1 = np.log(y1)
        dx = (x1 - x0) / (y1 - y0)
        fit = sinter.Fit(
            best=x1 + dx * (np.log(1e-12) - y1),
            low=low,
            high=high,
        )
    else:
        fit = sinter.fit_line_y_at_x(
            xs=[math.log(y) for y in ls_ys_fit],
            ys=ls_xs_fit,
            target_x=math.log(1e-12),
            max_extra_squared_error=0,
        )

    if fit.low < 0 or fit.best < 0 or fit.high < 0:
        return None
    return sinter.Fit(low=math.ceil(fit.low**2),
                      best=math.ceil(fit.best**2),
                      high=math.ceil(fit.high**2))


if __name__ == '__main__':
    main()
