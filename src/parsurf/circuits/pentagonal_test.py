import itertools

import pytest
import stim

from parsurf.circuits.chao_test import circuit_has_unsigned_stabilizers
from parsurf.circuits.pentagonal import iter_pentagonal_decompose_mpp4, \
    pentagonal_surface_code_memory_task
from parsurf.tools import Builder, AtLayer, not_nones


def test_pentagonal_mpp_x4_feedback():
    a, b, c, d, m1, m2 = range(6)
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder,
            use_classical_feedback=True):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'X': [c]}, {'X': [c]}, []),
        ({'X': [d]}, {'X': [d]}, []),
        ({'Z': [a, b]}, {'Z': [a, b]}, []),
        ({'Z': [b, c]}, {'Z': [b, c]}, []),
        ({'Z': [c, d]}, {'Z': [c, d]}, []),
        ({'X': [a, b, c, d]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_pentagonal_mpp_x4():
    a, b, c, d, m1, m2 = range(6)
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder,
            use_classical_feedback=False):
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


def test_pentagonal_mpp_x2_ab():
    a, b, c, d, m1, m2 = range(6)
    c = None
    d = None
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder,
            use_classical_feedback=False):
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


def test_pentagonal_mpp_x2_cd():
    a, b, c, d, m1, m2 = range(6)
    a = None
    b = None
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='X',
            builder=builder,
            use_classical_feedback=False):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('|', 'key'), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [c]}, {'X': [c]}, []),
        ({'X': [d]}, {'X': [d]}, []),
        ({'Z': [c, d]}, {'Z': [c, d]}, rec([k1, k2, k3])),
        ({'X': [c, d]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_pentagonal_mpp_z2_bc():
    a, b, c, d, m1, m2 = range(6)
    a = None
    d = None
    key = 'key'
    builder = Builder.for_qubits(not_nones([a, b, c, d, m1, m2]))
    for _ in iter_pentagonal_decompose_mpp4(
            a=a,
            b=b,
            c=c,
            d=d,
            m1=m1,
            m2=m2,
            key=key,
            layer=5,
            basis='Z',
            builder=builder,
            use_classical_feedback=False):
        pass

    rec = builder.tracker.current_measurement_record_targets_for
    k1 = AtLayer(m1, layer=5)
    k2 = AtLayer(m2, layer=5)
    k3 = AtLayer(('|', 'key'), layer=5)
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'Z': [b]}, {'Z': [b]}, []),
        ({'Z': [c]}, {'Z': [c]}, []),
        ({'X': [b, c]}, {'X': [b, c]}, rec([k1, k3])),
        ({'Z': [b, c]}, {}, rec([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


@pytest.mark.parametrize('diameter,basis,rounds,flip_orientation', itertools.product(
    range(3, 11),
    'XZ',
    [2, 3, 5, 10],
    [False, True],
))
def test_code_distance(diameter: int, basis: str, rounds: int, flip_orientation: bool):
    task = pentagonal_surface_code_memory_task(basis=basis, rounds=rounds, diam=diameter, noise=0.001, use_classical_feedback=False, flip_orientation=flip_orientation)

    task.circuit.detector_error_model()
    min_err = task.circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)

    assert len(min_err) == diameter - diameter // 2


def test_hooks_both_ways():
    task = pentagonal_surface_code_memory_task(basis='X', rounds=10, diam=7, use_classical_feedback=False, noise=0.001, flip_orientation=True)
    c2i = {tuple(v): k for k, v in task.circuit.get_detector_coordinates().items()}

    # Tile pitch is 4, so going from 6 to 14 (a gap of 8) with one error halves the code distance.
    a = c2i[(6, 6, 2)]
    b = c2i[(14, 6, 2)]
    c = c2i[(6, 14, 2)]

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


def test_exact_surface_code_memory_circuit():
    task = pentagonal_surface_code_memory_task(basis='X', rounds=100, diam=3, noise=0.001, flip_orientation=False)
    assert task.circuit == stim.Circuit("""
        QUBIT_COORDS(-1, 6) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(0, 4) 2
        QUBIT_COORDS(0, 8) 3
        QUBIT_COORDS(1, 2) 4
        QUBIT_COORDS(2, -1) 5
        QUBIT_COORDS(2, 5) 6
        QUBIT_COORDS(2, 7) 7
        QUBIT_COORDS(3, 2) 8
        QUBIT_COORDS(4, 0) 9
        QUBIT_COORDS(4, 4) 10
        QUBIT_COORDS(4, 8) 11
        QUBIT_COORDS(5, 6) 12
        QUBIT_COORDS(6, 1) 13
        QUBIT_COORDS(6, 3) 14
        QUBIT_COORDS(6, 9) 15
        QUBIT_COORDS(7, 6) 16
        QUBIT_COORDS(8, 0) 17
        QUBIT_COORDS(8, 4) 18
        QUBIT_COORDS(8, 8) 19
        QUBIT_COORDS(9, 2) 20
        RX 1 2 3 9 10 11 17 18 19
        R 0 4 8 12 16 20
        X_ERROR(0.001) 0 4 8 12 16 20
        Z_ERROR(0.001) 1 2 3 9 10 11 17 18 19
        DEPOLARIZE1(0.001) 5 6 7 13 14 15
        TICK
        MPP(0.001) X0*X2 X1*X4 X8*X9 X10*X12 X16*X18 X17*X20
        DEPOLARIZE2(0.001) 0 2 1 4 8 9 10 12 16 18 17 20
        DEPOLARIZE1(0.001) 3 5 6 7 11 13 14 15 19
        TICK
        MPP(0.001) Z4*Z8 Z12*Z16
        DEPOLARIZE2(0.001) 4 8 12 16
        DEPOLARIZE1(0.001) 0 1 2 3 5 6 7 9 10 11 13 14 15 17 18 19 20
        TICK
        RX 5 6 7 13 14 15
        MPP(0.001) X0*X3 X2*X4 X8*X10 X11*X12 X16*X19 X18*X20
        DEPOLARIZE2(0.001) 0 3 2 4 8 10 11 12 16 19 18 20
        Z_ERROR(0.001) 5 6 7 13 14 15
        DEPOLARIZE1(0.001) 1 9 17
        TICK
        M(0.001) 0 4 8 12 16 20
        MPP(0.001) Z1*Z5 Z2*Z6 Z3*Z7 Z9*Z13 Z10*Z14 Z11*Z15
        DEPOLARIZE1(0.001) 0 4 8 12 16 20
        DEPOLARIZE2(0.001) 1 5 2 6 3 7 9 13 10 14 11 15
        DEPOLARIZE1(0.001) 17 18 19
        TICK
        MPP(0.001) X6*X7 X13*X14
        DEPOLARIZE2(0.001) 6 7 13 14
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 8 9 10 11 12 15 16 17 18 19 20
        TICK
        R 0 4 8 12 16 20
        MPP(0.001) Z5*Z9 Z6*Z10 Z7*Z11 Z13*Z17 Z14*Z18 Z15*Z19
        DEPOLARIZE2(0.001) 5 9 6 10 7 11 13 17 14 18 15 19
        X_ERROR(0.001) 0 4 8 12 16 20
        DEPOLARIZE1(0.001) 1 2 3
        TICK
        MX(0.001) 5 6 7 13 14 15
        MPP(0.001) X0*X2 X1*X4 X8*X9 X10*X12 X16*X18 X17*X20
        DETECTOR(-2, 6, 0) rec[-46] rec[-38]
        DETECTOR(2, 2, 0) rec[-45] rec[-44] rec[-37] rec[-36]
        DETECTOR(6, 6, 0) rec[-43] rec[-42] rec[-35] rec[-34]
        DETECTOR(10, 2, 0) rec[-41] rec[-33]
        OBSERVABLE_INCLUDE(0) rec[-12] rec[-9]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 5 6 7 13 14 15
        DEPOLARIZE2(0.001) 0 2 1 4 8 9 10 12 16 18 17 20
        DEPOLARIZE1(0.001) 3 11 19
        TICK
        REPEAT 98 {
            MPP(0.001) Z4*Z8 Z12*Z16
            DEPOLARIZE2(0.001) 4 8 12 16
            DEPOLARIZE1(0.001) 0 1 2 3 5 6 7 9 10 11 13 14 15 17 18 19 20
            TICK
            RX 5 6 7 13 14 15
            MPP(0.001) X0*X3 X2*X4 X8*X10 X11*X12 X16*X19 X18*X20
            DEPOLARIZE2(0.001) 0 3 2 4 8 10 11 12 16 19 18 20
            Z_ERROR(0.001) 5 6 7 13 14 15
            DEPOLARIZE1(0.001) 1 9 17
            TICK
            M(0.001) 0 4 8 12 16 20
            MPP(0.001) Z1*Z5 Z2*Z6 Z3*Z7 Z9*Z13 Z10*Z14 Z11*Z15
            DEPOLARIZE1(0.001) 0 4 8 12 16 20
            DEPOLARIZE2(0.001) 1 5 2 6 3 7 9 13 10 14 11 15
            DEPOLARIZE1(0.001) 17 18 19
            TICK
            MPP(0.001) X6*X7 X13*X14
            DEPOLARIZE2(0.001) 6 7 13 14
            DEPOLARIZE1(0.001) 0 1 2 3 4 5 8 9 10 11 12 15 16 17 18 19 20
            TICK
            R 0 4 8 12 16 20
            MPP(0.001) Z5*Z9 Z6*Z10 Z7*Z11 Z13*Z17 Z14*Z18 Z15*Z19
            DEPOLARIZE2(0.001) 5 9 6 10 7 11 13 17 14 18 15 19
            X_ERROR(0.001) 0 4 8 12 16 20
            DEPOLARIZE1(0.001) 1 2 3
            TICK
            MX(0.001) 5 6 7 13 14 15
            MPP(0.001) X0*X2 X1*X4 X8*X9 X10*X12 X16*X18 X17*X20
            DETECTOR(-2, 6, 0) rec[-86] rec[-78] rec[-60] rec[-46] rec[-38]
            DETECTOR(2, -2, 0) rec[-66] rec[-58] rec[-40] rec[-26] rec[-18]
            DETECTOR(2, 2, 0) rec[-85] rec[-84] rec[-77] rec[-76] rec[-59] rec[-52] rec[-51] rec[-45] rec[-44] rec[-37] rec[-36]
            DETECTOR(2, 6, 0) rec[-65] rec[-64] rec[-57] rec[-56] rec[-40] rec[-32] rec[-31] rec[-30] rec[-29] rec[-25] rec[-24] rec[-17] rec[-16]
            DETECTOR(6, 2, 0) rec[-63] rec[-62] rec[-55] rec[-54] rec[-39] rec[-30] rec[-27] rec[-23] rec[-22] rec[-15] rec[-14]
            DETECTOR(6, 6, 0) rec[-83] rec[-82] rec[-75] rec[-74] rec[-60] rec[-51] rec[-50] rec[-48] rec[-47] rec[-43] rec[-42] rec[-35] rec[-34]
            DETECTOR(6, 10, 0) rec[-61] rec[-53] rec[-39] rec[-29] rec[-28] rec[-21] rec[-13]
            DETECTOR(10, 2, 0) rec[-81] rec[-73] rec[-59] rec[-49] rec[-48] rec[-41] rec[-33]
            OBSERVABLE_INCLUDE(0) rec[-12] rec[-9]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 5 6 7 13 14 15
            DEPOLARIZE2(0.001) 0 2 1 4 8 9 10 12 16 18 17 20
            DEPOLARIZE1(0.001) 3 11 19
            TICK
        }
        MPP(0.001) Z4*Z8 Z12*Z16
        DEPOLARIZE2(0.001) 4 8 12 16
        DEPOLARIZE1(0.001) 0 1 2 3 5 6 7 9 10 11 13 14 15 17 18 19 20
        TICK
        RX 5 6 7 13 14 15
        MPP(0.001) X0*X3 X2*X4 X8*X10 X11*X12 X16*X19 X18*X20
        DEPOLARIZE2(0.001) 0 3 2 4 8 10 11 12 16 19 18 20
        Z_ERROR(0.001) 5 6 7 13 14 15
        DEPOLARIZE1(0.001) 1 9 17
        TICK
        M(0.001) 0 4 8 12 16 20
        MPP(0.001) Z1*Z5 Z2*Z6 Z3*Z7 Z9*Z13 Z10*Z14 Z11*Z15
        DEPOLARIZE1(0.001) 0 4 8 12 16 20
        DEPOLARIZE2(0.001) 1 5 2 6 3 7 9 13 10 14 11 15
        DEPOLARIZE1(0.001) 17 18 19
        TICK
        MPP(0.001) X6*X7 X13*X14
        DEPOLARIZE2(0.001) 6 7 13 14
        DEPOLARIZE1(0.001) 0 1 2 3 4 5 8 9 10 11 12 15 16 17 18 19 20
        TICK
        MPP(0.001) Z5*Z9 Z6*Z10 Z7*Z11 Z13*Z17 Z14*Z18 Z15*Z19
        DEPOLARIZE2(0.001) 5 9 6 10 7 11 13 17 14 18 15 19
        DEPOLARIZE1(0.001) 0 1 2 3 4 8 12 16 20
        TICK
        MX(0.001) 5 6 7 13 14 15
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-3]
        MX(0.001) 1 2 3 9 10 11 17 18 19
        DETECTOR(-2, 6, 0) rec[-89] rec[-81] rec[-63] rec[-49] rec[-41]
        DETECTOR(2, -2, 0) rec[-69] rec[-61] rec[-43] rec[-29] rec[-21]
        DETECTOR(2, 2, 0) rec[-88] rec[-87] rec[-80] rec[-79] rec[-62] rec[-55] rec[-54] rec[-48] rec[-47] rec[-40] rec[-39]
        DETECTOR(2, 6, 0) rec[-68] rec[-67] rec[-60] rec[-59] rec[-43] rec[-35] rec[-34] rec[-33] rec[-32] rec[-28] rec[-27] rec[-20] rec[-19]
        DETECTOR(6, 2, 0) rec[-66] rec[-65] rec[-58] rec[-57] rec[-42] rec[-33] rec[-30] rec[-26] rec[-25] rec[-18] rec[-17]
        DETECTOR(6, 6, 0) rec[-86] rec[-85] rec[-78] rec[-77] rec[-63] rec[-54] rec[-53] rec[-51] rec[-50] rec[-46] rec[-45] rec[-38] rec[-37]
        DETECTOR(6, 10, 0) rec[-64] rec[-56] rec[-42] rec[-32] rec[-31] rec[-24] rec[-16]
        DETECTOR(10, 2, 0) rec[-84] rec[-76] rec[-62] rec[-52] rec[-51] rec[-44] rec[-36]
        SHIFT_COORDS(0, 0, 1)
        DETECTOR(-2, 6, 0) rec[-49] rec[-41] rec[-23] rec[-8] rec[-7]
        DETECTOR(2, 2, 0) rec[-48] rec[-47] rec[-40] rec[-39] rec[-22] rec[-15] rec[-14] rec[-9] rec[-8] rec[-6] rec[-5]
        DETECTOR(6, 6, 0) rec[-46] rec[-45] rec[-38] rec[-37] rec[-23] rec[-14] rec[-13] rec[-11] rec[-10] rec[-5] rec[-4] rec[-2] rec[-1]
        DETECTOR(10, 2, 0) rec[-44] rec[-36] rec[-22] rec[-12] rec[-11] rec[-3] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-9] rec[-6] rec[-3]
        DEPOLARIZE1(0.001) 5 6 7 13 14 15 1 2 3 9 10 11 17 18 19 0 4 8 12 16 20
    """)

