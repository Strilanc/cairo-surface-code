from typing import Optional, Iterator, Any, List, Iterable

import sinter
import stim

from parsurf.tools import Builder, AtLayer, NoiseModel, Tile, surface_code_tiles, not_nones


def iter_pentagonal_decompose_mpp4(
        *,
        a: Optional[complex],
        b: Optional[complex],
        c: Optional[complex],
        d: Optional[complex],
        m1: Optional[complex],
        m2: Optional[complex],
        key: Any,
        basis: str,
        builder: Builder,
        layer: int,
        use_classical_feedback: bool):
    """Iteratively appends operations into a builder, implementing a four body pauli product measurement.

    The intent is for a caller to start this generator alongside many others, and then
    incrementally advance them in synchronized steps in order to produce a circuit that measures
    all the tiles simultaneously.

    Args:
        builder: Where to put the operations.
        a: First data qubit (if present).
        b: Second data qubit (if present).
        c: Third data qubit (if present).
        d: Fourth data qubit (if present).
        m1: First measurement qubit.
        m2: Second measurement qubit.
        basis: The measurement basis.
        key: The key to store the measurement result under.
        layer: The time layer to store measurements keys under.
        use_classical_feedback: Whether or not classical feedback operations are used to make
            the parity measurements exactly correct, instead of requiring the detector and
            observable structure to change.

    Yields:
        A series of strings indicating what was just appended into the builder.
    """
    opp = 'Z' if basis == 'X' else 'X'

    builder.gate(f"R{opp}", not_nones([m1, m2]))
    yield 'R'

    builder.measure_pauli_product(qs={basis: not_nones([a, m1])}, key=('a', key), layer=layer)
    builder.measure_pauli_product(qs={basis: not_nones([b, m2])}, key=('b', key), layer=layer)
    yield 'AB'

    builder.measure_pauli_product(qs={opp: [m1, m2] * (m1 is not None and m2 is not None)}, key=('|', key), layer=layer)
    yield '|'

    builder.measure_pauli_product(qs={basis: not_nones([c, m1])}, key=('c', key), layer=layer)
    builder.measure_pauli_product(qs={basis: not_nones([d, m2])}, key=('d', key), layer=layer)
    yield 'CD'

    builder.measure(not_nones([m1, m2]), basis=opp, layer=layer)
    yield 'M'

    builder.tracker.make_measurement_group(
        [
            AtLayer(('a', key), layer),
            AtLayer(('b', key), layer),
            AtLayer(('c', key), layer),
            AtLayer(('d', key), layer),
        ],
        key=AtLayer(key, layer),
    )
    if use_classical_feedback:
        builder.classical_paulis(control_keys=[AtLayer(m1, layer)], targets=not_nones([c]), basis=basis)
        builder.classical_paulis(control_keys=[AtLayer(m2, layer)], targets=not_nones([d]), basis=basis)
        builder.classical_paulis(control_keys=[AtLayer(('|', key), layer)], targets=not_nones([a, c]), basis=basis)
    yield 'C'


def forever_iter_pentagonal_decompose_mpp_tile(
        *,
        tile: Tile,
        builder: Builder,
        use_classical_feedback: bool):
    a = tile.a
    b = tile.b
    c = tile.c
    d = tile.d
    m1 = tile.m1()
    m2 = tile.m2()

    layer = 0
    while True:
        yield from iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=tile.center,
            basis=tile.basis,
            builder=builder,
            layer=layer,
            use_classical_feedback=use_classical_feedback)
        layer += 1


def possible_tile_detector_keys(*, tile: Tile, layer: int, use_classical_feedback: bool) -> List[AtLayer]:
    c = tile.center
    possible_keys = [
        AtLayer(c, layer),
        AtLayer(c, layer - 1),
    ]
    if not use_classical_feedback:
        g = (lambda e: AtLayer(e, layer=layer)) if tile.basis == 'Z' else (lambda e: AtLayer(e, layer=layer - 1))
        f = (lambda e: c + e) if tile.um1().real == tile.um2().real else (lambda e: c + e.imag + 1j * e.real)
        possible_keys.extend([
            g(('|', f(-4j))),
            g(('|', f(4j))),
            g(f(-4j - 1)),
            g(f(-4j + 1)),
            g(f(3)),
            g(f(-3)),
        ])
    return possible_keys


def pentagonal_surface_code_memory_circuit(*, basis: str, rounds: int, diam: int, use_classical_feedback: bool = False, flip_orientation: bool) -> stim.Circuit:
    """Creates a stim circuit for a two-body measurement surface code memory experiment.

    Args:
        basis: The basis to initialize and measure the logical qubit in. Must be 'X' or 'Z'.
        rounds: How long the experiment preserves the logical qubit. The number of times each measurement qubit is
            measured (not counting two body measurements).
        diam: The width and height of the patch to prepare, in data qubits.
        use_classical_feedback: The decomposition of four body measurements into two body measurements produces
            classically controlled Paulis. They can be omitted by instead adjusting the definitions of detectors and
            observables. This option controls whether or not that is done. Note that noise is not applied to the
            classically controlled Paulis (they are assumed to be performed in the classical control system, not
            on the quantum computer.).
        flip_orientation: Changes the ordering used by the stabilizers.

    Returns:
        A noiseless circuit representing the experiment.
    """

    tiles = surface_code_tiles(diam=diam, flip_orientation=flip_orientation)
    data_set = {q for tile in tiles for q in tile.data_set}
    measure_set = {m for tile in tiles for m in tile.measure_set}
    used_set = data_set | measure_set
    builder = Builder.for_qubits(used_set)

    def append_partial_layer(expected: str, basis_iters: Iterable[Iterator[str]]):
        for it in basis_iters:
            ran_layer = next(it)
            if ran_layer != expected:
                raise ValueError(f"Expected layer {expected} but got {ran_layer}")

    def append_layers(xs: List[str], zs: List[str], z_first: bool = False, tick: bool = True):
        if z_first:
            for z in zs:
                append_partial_layer(z, z_iters)
        for x in xs:
            append_partial_layer(x, x_iters)
        if not z_first:
            for z in zs:
                append_partial_layer(z, z_iters)
        if tick:
            builder.tick()

    x_tiles = [tile for tile in tiles if tile.basis == 'X']
    z_tiles = [tile for tile in tiles if tile.basis == 'Z']
    x_iters = [forever_iter_pentagonal_decompose_mpp_tile(tile=tile, builder=builder, use_classical_feedback=use_classical_feedback) for tile in x_tiles]
    z_iters = [forever_iter_pentagonal_decompose_mpp_tile(tile=tile, builder=builder, use_classical_feedback=use_classical_feedback) for tile in z_tiles]

    builder.gate(f'R{basis}', data_set)

    append_layers(xs=['R'], zs=[])
    append_layers(xs=['AB'], zs=[])

    circuit_so_far = stim.Circuit()
    if use_classical_feedback:
        obs_feedback_keys = []
    else:
        relevant_coord = (lambda e: e.real) if basis == 'Z' else (lambda e: e.imag)
        if flip_orientation:
            obs_feedback_keys = [
                k
                for tile in tiles
                if tile.basis != basis
                and relevant_coord(tile.center) == -2
                for k in [('|', tile.center), tile.m1(), tile.m2()]
                if k is not None
            ] + [
                k
                for tile in tiles
                if tile.basis != basis
                and relevant_coord(tile.center) == 2
                for k in [('|', tile.center)]
                if k is not None
            ]
        else:
            obs_feedback_keys = [
                m
                for tile in tiles
                if tile.basis != basis
                for m in tile.measure_set
                if abs(relevant_coord(m)) == 1
            ]

    for layer in range(min(3, rounds) - 1):
        circuit_so_far += builder.circuit
        builder.circuit.clear()

        append_layers(xs=['|'], zs=[])
        append_layers(xs=['CD'], zs=['R'], z_first=True)
        append_layers(xs=['M', 'C'], zs=['AB'])
        append_layers(xs=[], zs=['|'])
        append_layers(xs=['R'], zs=['CD'])
        append_layers(xs=['AB'], zs=['M', 'C'], z_first=True, tick=False)
        for tile in tiles:
            if tile.basis != basis and layer == 0:
                continue
            builder.detector(
                possible_tile_detector_keys(tile=tile, layer=layer, use_classical_feedback=use_classical_feedback),
                pos=tile.center,
                ignore_non_existent=True)
        builder.obs_include([
            AtLayer(m, layer=layer)
            for m in obs_feedback_keys
        ], obs_index=0)
        builder.shift_coords(dt=1)
        builder.tick()

    if rounds >= 3:
        circuit_so_far += builder.circuit * (rounds - 2)
        builder.circuit.clear()
        last_layer = 2
    else:
        last_layer = rounds - 1

    append_layers(xs=['|'], zs=[])
    append_layers(xs=['CD'], zs=['R'], z_first=True)
    append_layers(xs=['M', 'C'], zs=['AB'])
    append_layers(xs=[], zs=['|'])
    append_layers(xs=[], zs=['CD'])
    append_layers(xs=[], zs=['M', 'C'], tick=False)
    builder.obs_include([
        AtLayer(m, layer=last_layer)
        for m in obs_feedback_keys
    ], obs_index=0)

    builder.measure(data_set, layer=last_layer, basis=basis)
    layer = last_layer
    for tile in tiles:
        builder.detector(
            possible_tile_detector_keys(tile=tile, layer=layer, use_classical_feedback=use_classical_feedback),
            pos=tile.center,
            ignore_non_existent=True)
    builder.shift_coords(dt=1)
    layer += 1
    for tile in tiles:
        if tile.basis == basis:
            builder.detector(
                possible_tile_detector_keys(tile=tile, layer=layer, use_classical_feedback=use_classical_feedback) + [AtLayer(q, layer - 1) for q in tile.data_set],
                pos=tile.center,
                ignore_non_existent=True)
    if basis == 'X':
        obs_qs = [q for q in data_set if q.imag == 0]
    else:
        obs_qs = [q for q in data_set if q.real == 0]
    builder.obs_include([AtLayer(q, last_layer) for q in obs_qs], obs_index=0)

    circuit_so_far += builder.circuit

    return circuit_so_far


def pentagonal_surface_code_memory_task(*, basis: str, rounds: int, diam: int, noise: float, use_classical_feedback: bool = False, flip_orientation: bool = False) -> sinter.Task:
    """Creates a sinter task for sampling from a two-body measurement surface code memory experiment.

    Args:
        basis: The basis to initialize and measure the logical qubit in. Must be 'X' or 'Z'.
        rounds: How long the experiment preserves the logical qubit. The number of times each measurement qubit is
            measured (not counting two body measurements).
        diam: The width and height of the patch to prepare, in data qubits.
        noise: How much noise to add into the circuit.
        use_classical_feedback: The decomposition of four body measurements into two body measurements produces
            classically controlled Paulis. They can be omitted by instead adjusting the definitions of detectors and
            observables. This option controls whether or not that is done. Note that noise is not applied to the
            classically controlled Paulis (they are assumed to be performed in the classical control system, not
            on the quantum computer.).
        flip_orientation: Changes the ordering used by the stabilizers.

    Returns:
        A sinter task representing the experiment.
    """
    circuit = pentagonal_surface_code_memory_circuit(basis=basis, rounds=rounds, diam=diam, use_classical_feedback=use_classical_feedback, flip_orientation=flip_orientation)
    noise_model = NoiseModel.depolarizing_two_body_measurement_noise(noise)
    noisy_circuit = noise_model.noisy_circuit(circuit)
    m = {
        'd': diam,
        'r': rounds,
        'b': basis,
        'p': noise,
        'c': 'pentagonal_smooth' if flip_orientation else 'pentagonal_sharp',
        'q': circuit.num_qubits,
    }
    if use_classical_feedback:
        m['use_classical_feedback'] = True

    return sinter.Task(
        circuit=noisy_circuit,
        json_metadata=m,
    )
