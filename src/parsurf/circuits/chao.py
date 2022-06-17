from typing import Any, Iterable, Iterator, List, Optional

import sinter
import stim

from parsurf.tools import Builder, AtLayer, NoiseModel, surface_code_tiles, Tile, not_nones


def iter_chao_decompose_mpp4(
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
    builder.gate(f"R{opp}", [m1])
    yield 'R'

    builder.measure_pauli_product(qs={bas: not_nones([a, m1])}, key=('a_m1', a, m1), layer=layer)
    yield 'P1_a'
    builder.classical_paulis(control_keys=[AtLayer(('a_m1', a, m1), layer=layer)], targets=[m1], basis=opp)
    yield 'P1_b'
    builder.gate(f"R{bas}", [m2])
    yield 'P1_c'
    builder.measure_pauli_product(qs={opp: [m1, m2]}, key=('m1_m2_1', m1, m2), layer=layer)
    yield 'P2_a'
    builder.classical_paulis(control_keys=[AtLayer(('m1_m2_1', m1, m2), layer=layer)], targets=not_nones([a, m1]), basis=bas)
    yield 'P2_b'
    builder.measure_pauli_product(qs={bas: not_nones([b, m1])}, key=('b_m1', b, m1), layer=layer)
    yield 'P3_a'
    builder.classical_paulis(control_keys=[AtLayer(('b_m1', b, m1), layer=layer)], targets=[m1, m2], basis=opp)
    yield 'P3_b'

    builder.measure([m1], basis=opp, tracker_key=lambda _: ('anc1', m1), layer=layer)
    yield 'M1_a'
    builder.classical_paulis(control_keys=[AtLayer(('anc1', m1), layer=layer)], targets=not_nones([b]), basis=bas)
    yield 'M1_b'
    builder.gate(f"R{opp}", [m1])
    yield 'R3'

    builder.measure_pauli_product(qs={bas: not_nones([c, m1])}, key=('c_m1', c, m1), layer=layer)
    yield 'P4_a'
    builder.classical_paulis(control_keys=[AtLayer(('c_m1', c, m1), layer=layer)], targets=[m1], basis=opp)
    yield 'P4_b'
    builder.measure_pauli_product(qs={opp: [m1, m2]}, key=('m1_m2_2', m1, m2), layer=layer)
    yield 'P5_a'
    builder.classical_paulis(control_keys=[AtLayer(('m1_m2_2', m1, m2), layer=layer)], targets=not_nones([c, m1]), basis=bas)
    yield 'P5_b'
    builder.measure_pauli_product(qs={bas: not_nones([d, m1])}, key=('d_m1', d, m1), layer=layer)
    yield 'P6_a'
    builder.measure([m2], basis=bas, tracker_key=lambda _: ('anc2', m2), layer=layer)
    yield 'P6_b'

    builder.measure([m1], basis=opp, tracker_key=lambda _: ('anc2', m1), layer=layer)
    yield 'M2_a'
    builder.classical_paulis(control_keys=[AtLayer(('anc2', m1), layer=layer)], targets=not_nones([d]), basis=bas)
    yield 'M2_b'
    builder.tracker.make_measurement_group([
        AtLayer(('d_m1', d, m1), layer=layer),
        AtLayer(('anc2', m2), layer=layer),
    ], key=AtLayer(key, layer=layer))
    yield 'C'


def forever_iter_chao_decompose_mpp_tile(
        *,
        tile: Tile,
        builder: Builder):
    a = tile.a
    b = tile.c
    c = tile.b
    d = tile.d

    layer = 0
    while True:
        yield from iter_chao_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=tile.center,
            m2=tile.center + 1,
            key=tile.center, basis=tile.basis, builder=builder, layer=layer)
        layer += 1


def chao_memory_experiment_circuit(*, diam: int, basis: str, rounds: int) -> stim.Circuit:
    tiles = surface_code_tiles(diam=diam, flip_orientation=False)
    data_set = {q for tile in tiles for q in tile.data_set}
    measure_set = {tile.center for tile in tiles} | {tile.center + 1 for tile in tiles}
    used_set = data_set | measure_set
    builder = Builder.for_qubits(used_set)

    iter_xs = []
    iter_zs = []
    for tile in tiles:
        it = forever_iter_chao_decompose_mpp_tile(tile=tile, builder=builder)
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

    builder.gate(f"R{basis}", data_set)
    circuit_so_far = stim.Circuit()
    for layer in range(min(2, rounds)):
        circuit_so_far += builder.circuit
        builder.circuit.clear()
        append_layers(['R'])
        append_layers(['P1_a', 'P1_b', 'P1_c'])
        append_layers(['P2_a', 'P2_b'])
        append_layers(['P3_a', 'P3_b'])
        append_layers(['M1_a', 'M1_b'])
        append_layers(['R3'])
        append_layers(['P4_a', 'P4_b'])
        append_layers(['P5_a', 'P5_b'])
        append_layers(['P6_a', 'P6_b'])
        append_layers(['M2_a', 'M2_b', 'C'], tick=False)
        for tile in tiles:
            if layer == 0:
                if tile.basis == basis:
                    builder.detector([AtLayer(tile.center, layer=layer)], pos=tile.center)
            else:
                builder.detector([AtLayer(tile.center, layer=layer), AtLayer(tile.center, layer=layer - 1)], pos=tile.center)
        builder.shift_coords(dt=1)
        builder.tick()

    circuit_so_far += builder.circuit * max(1, rounds - 1)
    builder.circuit.clear()

    last_layer = min(1, rounds - 1)
    builder.measure(data_set, basis=basis, layer=last_layer)
    for tile in tiles:
        if tile.basis == basis:
            builder.detector([AtLayer(q, layer=last_layer) for q in [*tile.data_set, tile.center]], pos=tile.center)

    if basis == 'X':
        obs_qubits = [d for d in data_set if d.imag == 0]
    elif basis == 'Z':
        obs_qubits = [d for d in data_set if d.real == 0]
    else:
        raise NotImplementedError(f'{basis=}')

    builder.obs_include([AtLayer(d, layer=last_layer) for d in obs_qubits], obs_index=0)

    return circuit_so_far + builder.circuit


def chao_memory_experiment_task(*, basis: str, rounds: int, diam: int, noise: float) -> sinter.Task:
    circuit = chao_memory_experiment_circuit(basis=basis, rounds=rounds, diam=diam)
    noise_model = NoiseModel.depolarizing_two_body_measurement_noise(noise)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    m = {
        'd': diam,
        'r': rounds,
        'b': basis,
        'p': noise,
        'c': 'chao',
        'q': circuit.num_qubits,
    }

    return sinter.Task(
        circuit=noisy_circuit,
        json_metadata=m,
    )
