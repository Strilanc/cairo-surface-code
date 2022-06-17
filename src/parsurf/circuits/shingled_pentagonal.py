from typing import Any, Iterable, Iterator, List, Optional

import sinter
import stim

from parsurf.tools import Builder, AtLayer, NoiseModel, surface_code_tiles, Tile, not_nones


def iter_shingled_pentagonal_decompose_mpp4(
        *,
        a: Optional[complex],
        b: Optional[complex],
        c: Optional[complex],
        d: Optional[complex],
        m1: complex,
        m2: complex,
        key: Any,
        basis: str,
        builder: Builder, layer: int):
    bas = basis
    opp = 'Z' if bas == 'X' else 'X'

    builder.gate(f"R{opp}", not_nones([m1, m2]))
    yield 'R'
    builder.measure_pauli_product(qs={bas: not_nones([a, m1])}, key=('a', key), layer=layer)
    yield 'A'
    builder.measure_pauli_product(qs={bas: not_nones([b, m2])}, key=('b', key), layer=layer)
    yield 'B'
    builder.measure_pauli_product(qs={opp: [m1, m2] * (m1 is not None and m2 is not None)}, key=('|', key), layer=layer)
    yield '|'
    builder.measure_pauli_product(qs={bas: not_nones([c, m1])}, key=('c', key), layer=layer)
    yield 'C'
    builder.measure_pauli_product(qs={bas: not_nones([d, m2])}, key=('d', key), layer=layer)
    yield 'D'
    builder.measure(not_nones([m1, m2]), basis=opp, layer=layer)
    yield 'M'
    builder.tracker.make_measurement_group([
        AtLayer(('a', key), layer=layer),
        AtLayer(('b', key), layer=layer),
        AtLayer(('c', key), layer=layer),
        AtLayer(('d', key), layer=layer),
    ], key=AtLayer(key, layer))
    yield 'C'


def forever_iter_shingled_pentagonal_decompose_mpp_tile(
        *,
        tile: Tile,
        builder: Builder):
    a = tile.a
    b = tile.b
    c = tile.c
    d = tile.d
    m1 = tile.m1()
    m2 = tile.m2()

    layer = 0
    while True:
        yield from iter_shingled_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=tile.center, basis=tile.basis, builder=builder, layer=layer)
        layer += 1


def shingled_pentagonal_memory_experiment_circuit(*, diam: int, basis: str, rounds: int) -> stim.Circuit:
    tiles = surface_code_tiles(diam=diam, flip_orientation=False)
    data_set = {d for tile in tiles for d in tile.data_set}
    measure_set = {m for tile in tiles for m in tile.measure_set}
    used_set = data_set | measure_set
    builder = Builder.for_qubits(used_set)

    iter_xs = []
    iter_zs = []
    for tile in tiles:
        it = forever_iter_shingled_pentagonal_decompose_mpp_tile(tile=tile, builder=builder)
        if tile.basis == 'X':
            iter_xs.append(it)
        else:
            iter_zs.append(it)

    def append_partial_layer(expected: str, basis_iters: Iterable[Iterator[str]]):
        for it in basis_iters:
            ran_layer = next(it)
            if ran_layer != expected:
                raise ValueError(f"Expected layer {expected} but got {ran_layer}")

    def append_layers(layers: List[str], tick: bool = True):
        for x in layers:
            append_partial_layer(x, iter_xs)
        for z in layers:
            append_partial_layer(z, iter_zs)
        if tick:
            builder.tick()
    existing_tiles = {tile.center for tile in tiles}

    if basis == 'X':
        obs_qubits = [d for d in data_set if d.imag == 0]
        obs_anticomms = [m for m in measure_set if abs(m.imag) == 1]
    elif basis == 'Z':
        obs_qubits = [d for d in data_set if d.real == 0]
        obs_anticomms = [m for m in measure_set if abs(m.real) == 1]
    else:
        raise NotImplementedError(f'{basis=}')

    builder.gate(f"R{basis}", data_set)
    for layer in range(rounds):
        append_layers(['R'])
        append_layers(['A'])
        append_layers(['B'])
        append_layers(['|'])
        append_layers(['C'])
        append_layers(['D'])
        append_layers(['M', 'C'], tick=False)
        for tile in tiles:
            if layer != 0 or basis == tile.basis:
                f = (lambda e: tile.center + e) if tile.basis == 'Z' else (lambda e: tile.center + e.imag + e.real * 1j)
                cur = lambda e: AtLayer(e, layer=layer)
                prev = lambda e: AtLayer(e, layer=layer - 1)
                builder.detector([
                    cur(f(0)),
                    *([prev(f(0))] * (layer > 0)),
                    *([
                          prev(f(-3)),
                    ] * (f(-4) in existing_tiles and layer > 0)),
                    *([
                          cur(f(3)),
                      ] * (f(4) in existing_tiles)),
                    *([
                        prev(('|', f(-4j))),
                        prev(f(-4j - 1)),
                        prev(f(-4j + 1)),
                    ] * (f(-4j) in existing_tiles and layer > 0)),
                    *([
                        cur(('|', f(4j))),
                    ] * (f(4j) in existing_tiles)),
                ], pos=tile.center)
        builder.shift_coords(dt=1)
        builder.obs_include([AtLayer(m, layer=layer) for m in obs_anticomms], obs_index=0)
        if layer < rounds - 1:
            builder.tick()

    builder.measure(data_set, basis=basis, layer=rounds - 1)
    for tile in tiles:
        if tile.basis == basis:
            layer = rounds
            f = (lambda e: tile.center + e) if tile.basis == 'Z' else (lambda e: tile.center + e.imag + e.real * 1j)
            prev = lambda e: AtLayer(e, layer=layer - 1)
            builder.detector([
                prev(f(0)),
                *[prev(q) for q in tile.data_set],
                *([
                      prev(f(-3)),
                  ] * (f(-4) in existing_tiles)),
                *([
                      prev(('|', f(-4j))),
                      prev(f(-4j - 1)),
                      prev(f(-4j + 1)),
                  ] * (f(-4j) in existing_tiles)),
            ], pos=tile.center)

    builder.obs_include([AtLayer(d, layer=rounds - 1) for d in obs_qubits], obs_index=0)

    return builder.circuit


def shingled_pentagonal_memory_task(*, basis: str, rounds: int, diam: int, noise: float) -> sinter.Task:
    circuit = shingled_pentagonal_memory_experiment_circuit(basis=basis, rounds=rounds, diam=diam)
    noise_model = NoiseModel.depolarizing_two_body_measurement_noise(noise)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    m = {
        'd': diam,
        'r': rounds,
        'b': basis,
        'p': noise,
        'c': 'shingled_pentagonal',
        'q': circuit.num_qubits,
    }

    return sinter.Task(
        circuit=noisy_circuit,
        json_metadata=m,
    )
