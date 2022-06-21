#!/usr/bin/env python3

import argparse
import math
import pathlib
from typing import List, Callable

import sinter
from matplotlib import pyplot as plt
import matplotlib

from parsurf.scripts.plot_extrapolation import TITLE_CHANGE, TITLE_ORDER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, nargs='+', type=str)
    parser.add_argument("--chunking", required=True, choices=['shot', 'd', 'round'])
    parser.add_argument("--skip_d", default=(), nargs='+', type=int)
    parser.add_argument("--skip_p", default=(), nargs='+', type=float)
    parser.add_argument("--skip_b", default=(), nargs='+', type=str)
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
    x_func = lambda stat: stat.json_metadata['p']

    samples = sinter.stats_from_csv_files(*args.csv)
    samples = [e for e in samples
               if e.json_metadata['d'] not in args.skip_d
               and e.json_metadata['p'] not in args.skip_p
               and e.json_metadata['b'] not in args.skip_b]

    def curve_func(stat: sinter.TaskStats) -> str:
        m = stat.json_metadata
        return f"d=width={m['d']}, rounds={m['r']}"

    color_key = curve_func
    marker_key = curve_func
    color_index = {t: k for k, t in enumerate(sorted({color_key(e) for e in samples}, key=sinter.better_sorted_str_terms))}
    marker_index = {t: k for k, t in enumerate(sorted({marker_key(e) for e in samples}, key=sinter.better_sorted_str_terms))}

    group_func = lambda e: e.json_metadata['c']
    groups = sinter.group_by(samples, key=group_func)
    fig, axs = plt.subplots(1, len(groups), gridspec_kw={'wspace': 0.05, 'hspace': 0})
    for k, key in enumerate(sorted(groups.keys(), key=TITLE_ORDER.__getitem__)):
        ax: plt.Axes = axs[k]
        plot_case(
            ax=ax,
            samples=groups[key],
            x_func=x_func,
            marker_func=lambda e: MARKERS[marker_index[marker_key(e)]],
            color_func=lambda e: COLORS[color_index[color_key(e)]],
            curve_func=curve_func,
            s2p=s2p,
        )
        ax.set_title(TITLE_CHANGE[key], fontsize=10)
        ax.loglog()
        ax.set_ylabel(f'Logical Error Rate (per {per_name})')
        ax.set_xlabel('Physical Error Rate')
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.set_yticks([10 ** d for d in range(-12, 1)])
        ax.set_yticks([b * 10 ** d for d in range(-12, 0) for b in range(2, 10)], minor=True)
        if k > 0:
            ax.set_yticklabels([])
            ax.set_yticklabels([], minor=True)
        ax.set_xticks([10 ** d for d in range(-12, 1)])
        ax.set_xticks([b * 10 ** d for d in range(-12, 0) for b in range(2, 10)], minor=True)
        ax.set_xticklabels(["$10^{" + str(d) + "}$" for d in range(-12, 1)], rotation=70)

        ax.set_xlim(1e-4, 1e-2)
        ax.set_ylim(1e-8, 1)
        ax.grid(which='major', c='black')
        ax.grid(which='minor')

        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.yaxis.label.set_visible(k == 0)
    axs[-1].legend(
        handles=[
            matplotlib.lines.Line2D([0], [0], lw=0, label="Size")

        ] + [
            matplotlib.lines.Line2D([0], [0], color=COLORS[color_index[k]], lw=1, linestyle='dashed', marker=MARKERS[marker_index[k]], label=str(k))
            for k in sorted({curve_func(e) for e in samples}, key=sinter.better_sorted_str_terms)
        ],
        loc="upper left",
        prop={'size': 12},
        bbox_to_anchor=(1, 1),
    )

    fig.set_size_inches(16, 9)
    if args.save is not None:
        pathlib.Path(args.save).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.save, bbox_inches='tight', dpi=256)
        print(f"wrote {args.save}")
    if args.show:
        plt.show()


def plot_case(
        *,
        ax: plt.Axes,
        samples: List[sinter.TaskStats],
        x_func: Callable[[sinter.TaskStats], float],
        marker_func: Callable[[sinter.TaskStats], str],
        color_func: Callable[[sinter.TaskStats], str],
        curve_func: Callable[[sinter.TaskStats], str],
        s2p: Callable[[float, sinter.TaskStats], float],
) -> None:
    curves = sinter.group_by(samples, key=curve_func)
    for label in sorted(curves.keys(), key=sinter.better_sorted_str_terms):
        curve_stats = sorted(curves[label], key=x_func)
        color = color_func(curve_stats[0])
        marker = marker_func(curve_stats[0])

        xs = []
        ys = []
        xs_range = []
        ys_low = []
        ys_high = []
        for stat in curve_stats:
            fit = sinter.fit_binomial(
                num_shots=stat.shots,
                num_hits=stat.errors,
                max_likelihood_factor=1000,
            )
            x = x_func(stat)

            if stat.errors:
                xs.append(x)
                ys.append(s2p(fit.best, stat))

            xs_range.append(x)
            ys_low.append(s2p(fit.low, stat))
            ys_high.append(s2p(fit.high, stat))
        ax.plot(xs, ys, label=curve_func(curve_stats[0]), marker=marker, color=color)
        ax.fill_between(xs_range, ys_low, ys_high, alpha=0.25, color=color)


if __name__ == '__main__':
    main()
