#!/usr/bin/env python3

import argparse
import math

from parsurf.circuits.pentagonal import possible_tile_detector_keys
from parsurf.tools import surface_code_tiles


def pentagonal_surface_code_svg(*, diam: int, show_order: bool = False, show_feed: bool = False, flip_orientation: bool) -> str:
    tiles = surface_code_tiles(diam=diam, flip_orientation=flip_orientation)
    lines = []
    canvas_width = 512
    canvas_height = 512
    used_set = {q for tile in tiles for q in tile.used_set}
    measure_set = {q for tile in tiles for q in tile.measure_set}
    data_set = {q for tile in tiles for q in tile.data_set}
    min_r = min(q.real for q in used_set)
    min_i = min(q.imag for q in used_set)
    max_r = max(q.real for q in used_set)
    max_i = max(q.imag for q in used_set)
    min_c = min_r + min_i*1j
    max_c = max_r + max_i*1j
    pad = 5
    min_c -= (1 + 1j) * pad
    max_c += (1 + 1j) * pad
    scale = max((max_c.real - min_c.real) / canvas_width, (max_c.imag - min_c.imag) / canvas_height)

    BACKGROUND_X = '#FFDDDD'
    BACKGROUND_Z = '#DDDDFF'
    BACKGROUND_STROKE = '#DDDDDD'
    FOREGROUND_X = '#FF0000'
    FOREGROUND_Z = '#0000FF'
    PARITY_STROKE_WIDTH = 1 if show_feed or show_order else 3
    QUBIT_RADIUS = 1 if show_feed else 4
    FEED_STROKE_WIDTH = 4
    FEED_QUBIT_RADIUS = 10

    def pt0(c: complex) -> complex:
        c -= min_c
        c /= scale
        return c

    def pt(c: complex) -> str:
        c = pt0(c)
        return f'{c.real},{c.imag}'

    def dt(d: complex) -> str:
        d = pt0(d) - pt0(0)
        return f'{d.real},{d.imag}'

    lines.append(f"""<svg viewBox="0 0 {canvas_width} {canvas_height}" xmlns="http://www.w3.org/2000/svg">""")
    for tile in tiles:
        background = BACKGROUND_X if tile.basis == 'X' else BACKGROUND_Z

        if len(tile.data_set) == 2:
            a, b = tile.data_set
            da = a - tile.center
            db = b - tile.center
            dab = math.atan2(da.imag, da.real) - math.atan2(db.imag, db.real)
            dab %= math.pi * 2
            if dab < math.pi:
                a, b = b, a
            lines.append(f'<path d="M{pt(a)} a1,1 0 0,0 {dt(b - a)} M {pt(a)} {pt(b)}" fill="{background}" stroke="{BACKGROUND_STROKE}" />')
        else:
            qs = sorted(tile.data_set, key=lambda p2: math.atan2(p2.imag - tile.center.imag, p2.real - tile.center.real))
            x = f'<path d="M{pt(qs[-1])}'
            for q in qs:
                x += ' ' + pt(q)
            x += '"'
            lines.append(f'{x} fill="{background}" stroke="{BACKGROUND_STROKE}" />')

        data_stroke = FOREGROUND_X if tile.basis == 'X' else FOREGROUND_Z
        mm_stroke = FOREGROUND_Z if tile.basis == 'X' else FOREGROUND_X
        for k, a in enumerate([tile.a, tile.b, tile.c, tile.d]):
            if a is None:
                continue
            t = 1 if k in [0, 1] else 3
            if tile.basis == 'X':
                t += 3
            b = tile.m1() if k in [0, 2] else tile.m2()
            c = pt0((a + b) / 2)
            lines.append(
                f'<path d="M {pt(a)} {pt(b)}" stroke="{data_stroke}" stroke-width="{PARITY_STROKE_WIDTH}"/>')
            if show_order:
                lines.append(
                    f'<text x="{c.real}" y="{c.imag}" fill="black" font-size="20" text-anchor="middle" alignment-baseline="central">{t}</text>')

        if len(tile.measure_set) == 2:
            t = 2
            if tile.basis == 'X':
                t += 3
            a, b = tile.measure_set
            c = pt0((a + b) / 2)
            lines.append(
                f'<path d="M {pt(a)} {pt(b)}" stroke="{mm_stroke}" stroke-width="{PARITY_STROKE_WIDTH}" />')
            if show_order:
                lines.append(
                    f'<text x="{c.real}" y="{c.imag}" fill="black" font-size="20" text-anchor="middle" alignment-baseline="central">{t}</text>')
        for m in tile.measure_set:
            t = 0
            if tile.basis == 'X':
                t += 3
            c = pt0(tile.center + (m - tile.center) * 1.5)
            if show_order:
                lines.append(
                    f'<text x="{c.real}" y="{c.imag}" fill="black" font-size="8" text-anchor="middle" alignment-baseline="central">{t},{t+4}</text>')
    for q in measure_set:
        lines.append(f'<circle cx="{pt0(q).real}" cy="{pt0(q).imag}" r="{QUBIT_RADIUS}" fill="black" stroke="black" />')
    for q in data_set:
        lines.append(f'<circle cx="{pt0(q).real}" cy="{pt0(q).imag}" r="{QUBIT_RADIUS}" fill="white" stroke="black" />')

    q2t = {tile.center: tile for tile in tiles}
    if show_feed:
        for tile in tiles:
            if tile.center == 10 + 10j:
                for key in possible_tile_detector_keys(tile=tile, layer=0, use_classical_feedback=False):
                    if isinstance(key.key, complex):
                        if key.key == tile.center:
                            for a, b in [(tile.a, tile.m1()), (tile.b, tile.m2()), (tile.c, tile.m1()), (tile.d, tile.m2())]:
                                if a is not None:
                                    lines.append(
                                        f'<path d="M {pt(a)} {pt(b)}" stroke="white" stroke-width="{FEED_STROKE_WIDTH + 6}" />')
                                    lines.append(
                                        f'<path d="M {pt(a)} {pt(b)}" stroke="black" stroke-width="{FEED_STROKE_WIDTH}" />')

                        else:
                            a = key.key
                            k = pt0(a)
                            lines.append(
                                f'<circle cx="{k.real}" cy="{k.imag}" r="{FEED_QUBIT_RADIUS}" fill="black" stroke="white" stroke-width="3"/>')
                    else:
                        t, o = key.key
                        other = q2t[o]
                        if t == '|':
                            a, b = other.um1(), other.um2()
                        else:
                            raise NotImplementedError()
                        lines.append(
                            f'<path d="M {pt(a)} {pt(b)}" stroke="white" stroke-width="{FEED_STROKE_WIDTH + 6}" />')
                        lines.append(
                            f'<path d="M {pt(a)} {pt(b)}" stroke="black" stroke-width="{FEED_STROKE_WIDTH}" />')

    lines.append("</svg>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--diam", required=True, type=int)
    parser.add_argument('--show_order', action='store_true')
    parser.add_argument('--show_feed', action='store_true')
    parser.add_argument('--flip_orientation', action='store_true')
    args = parser.parse_args()
    svg = pentagonal_surface_code_svg(diam=args.diam, show_order=args.show_order, show_feed=args.show_feed, flip_orientation=args.flip_orientation)
    if not args.out:
        print(svg)
    else:
        with open(args.out, 'w') as f:
            print(svg, file=f)


if __name__ == '__main__':
    main()
