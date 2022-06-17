import itertools

import pytest
import stim

from parsurf.tools import Builder, AtLayer, not_nones
from parsurf.circuits.chao_test import circuit_has_unsigned_stabilizers
from parsurf.circuits.shingled_pentagonal import iter_shingled_pentagonal_decompose_mpp4, \
    shingled_pentagonal_memory_task


def test_pentagonal_mpp_x4():
    a, b, c, d, m1, m2 = range(6)
    key = 'key'
    builder = Builder.for_qubits([a, b, c, d, m1, m2])
    for _ in iter_shingled_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('|', 'key'), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'X': [c]}, {'X': [c]}, []),
        ({'X': [d]}, {'X': [d]}, []),
        ({'Z': [a, b]}, {'Z': [a, b]}, rec([k3])),
        ({'Z': [b, c]}, {'Z': [b, c]}, rec([k1, k3])),
        ({'Z': [c, d]}, {'Z': [c, d]}, rec([k1, k2, k3])),
        ({'X': [a, b, c, d]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_pentagonal_mpp_x2():
    a, b, m1, m2 = range(4)
    c = None
    d = None
    key = 'key'
    builder = Builder.for_qubits([a, b, m1, m2])
    for _ in iter_shingled_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('|', 'key'), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'Z': [a, b]}, {'Z': [a, b]}, rec([k3])),
        ({'X': [a, b]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_pentagonal_mpp_z2():
    c, d, m1, m2 = range(4)
    a = None
    b = None
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_shingled_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='Z',
            builder=builder):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('|', 'key'), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'Z': [c]}, {'Z': [c]}, []),
        ({'Z': [d]}, {'Z': [d]}, []),
        ({'X': [c, d]}, {'X': [c, d]}, rec([k1, k2, k3])),
        ({'Z': [c, d]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_pentagonal_mpp_z2_cross():
    b, d, m1, m2 = range(4)
    a = None
    c = None
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_shingled_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='Z',
            builder=builder):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('m1_m2', m1, m2), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'Z': [b]}, {'Z': [b]}, []),
        ({'Z': [d]}, {'Z': [d]}, []),
        ({'X': [b, d]}, {'X': [b, d]}, rec([k2])),
        ({'Z': [b, d]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


@pytest.mark.parametrize('diameter,basis,rounds', itertools.product(
    range(3, 11),
    'XZ',
    [2, 3, 5, 10],
))
def test_code_distance(diameter: int, basis: str, rounds: int):
    task = shingled_pentagonal_memory_task(basis=basis, rounds=rounds, diam=diameter, noise=0.001)

    task.circuit.detector_error_model()
    min_err = task.circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)

    assert len(min_err) == diameter - min(rounds, diameter // 2)


def test_hooks_both_ways():
    task = shingled_pentagonal_memory_task(basis='X', rounds=10, diam=7, noise=0.001)
    c2i = {tuple(v): k for k, v in task.circuit.get_detector_coordinates().items()}

    # Tile pitch is 4, so going from 6 to 14 (a gap of 8) with one error halves the code distance.
    a = c2i[(6, 6, 2)]
    b = c2i[(14, 6, 3)]
    c = c2i[(6, 14, 3)]

    ab, = task.circuit.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(f"error(1) D{a} D{b}"),
        reduce_to_one_representative_error=True,
    )
    bc, = task.circuit.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel(f"error(1) D{a} D{c}"),
        reduce_to_one_representative_error=True,
    )
    assert len(ab.circuit_error_locations) == 1
    assert len(bc.circuit_error_locations) == 1
