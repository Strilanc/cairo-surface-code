#!/usr/bin/env python3

import argparse
import pathlib
import sys

from parsurf.circuits.chao import chao_memory_experiment_task
from parsurf.circuits.pentagonal import pentagonal_surface_code_memory_task
from parsurf.circuits.ref_honeycomb import generate_honeycomb_task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--basis", nargs='+', required=True, type=str)
    parser.add_argument("--noise", nargs='+', required=True, type=float)
    parser.add_argument("--round_factors", nargs='+', required=True, type=int)
    parser.add_argument("--diam", nargs='+', required=True, type=int)
    parser.add_argument('--use_classical_feedback', action='store_true')
    parser.add_argument('--honeycomb', required=True, type=int)
    args = parser.parse_args()

    methods = [
        chao_memory_experiment_task,
        pentagonal_surface_code_memory_task,
    ]
    if args.honeycomb != 0:
        methods.append(generate_honeycomb_task)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    for basis in args.basis:
        for noise in args.noise:
            for diam in args.diam:
                for round_factor in args.round_factors:
                    for method in methods:
                        rounds = round_factor * diam
                        task = method(
                            basis=basis,
                            rounds=rounds,
                            diam=diam,
                            noise=noise)
                        m = task.json_metadata
                        name = ','.join(f'{k}={m[k]}' for k in sorted(m.keys()))
                        path = out_dir / f'{name}.stim'
                        with open(path, 'w') as f:
                            print(task.circuit, file=f)
                        print(f'wrote {path}', file=sys.stderr)


if __name__ == '__main__':
    main()
