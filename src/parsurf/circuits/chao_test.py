import itertools

import pytest
import stim

from parsurf.circuits.chao import iter_chao_decompose_mpp4, \
    chao_memory_experiment_task
from parsurf.tools import Builder, AtLayer, circuit_has_unsigned_stabilizers


def test_chao_decompose_mxx4():
    a, b, c, d = 0, 1, 2, 3
    m1 = 4
    m2 = 5
    key = 'key'
    builder = Builder.for_qubits([a, b, c, d, m1, m2])
    for _ in iter_chao_decompose_mpp4(
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
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'X': [c]}, {'X': [c]}, []),
        ({'X': [d]}, {'X': [d]}, []),

        ({'Z': [a, b]}, {'Z': [a, b]}, []),
        ({'Z': [b, c]}, {'Z': [b, c]}, []),
        ({'Z': [c, d]}, {'Z': [c, d]}, []),
        ({'X': [a, b, c, d]}, {}, builder.tracker.current_measurement_record_targets_for([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_chao_decompose_mzz4():
    a, b, c, d = 0, 1, 2, 3
    m1 = 1j
    m2 = 2j
    key = 'key'
    builder = Builder.for_qubits([a, b, c, d, m1, m2])
    for _ in iter_chao_decompose_mpp4(
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
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'Z': [a]}, {'Z': [a]}, []),
        ({'Z': [b]}, {'Z': [b]}, []),
        ({'Z': [c]}, {'Z': [c]}, []),
        ({'Z': [d]}, {'Z': [d]}, []),
        ({'X': [a, b]}, {'X': [a, b]}, []),
        ({'X': [b, c]}, {'X': [b, c]}, []),
        ({'X': [c, d]}, {'X': [c, d]}, []),
        ({'Z': [a, b, c, d]}, {}, builder.tracker.current_measurement_record_targets_for([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_chao_decompose_mxx2():
    a, b = 0, 1
    m = 1j
    key = 'key'
    builder = Builder.for_qubits([a, b, m, m + 1])
    for _ in iter_chao_decompose_mpp4(
            a=a,
            b=b,
            c=None,
            d=None,
            m1=m,
            m2=m + 1,
            key=key,
            layer=5,
            basis='X',
            builder=builder):
        pass
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'Z': [a, b]}, {'Z': [a, b]}, []),
        ({'X': [a, b]}, {}, builder.tracker.current_measurement_record_targets_for([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_chao_decompose_mxx2b():
    a, b = 0, 1
    m = 1j
    key = 'key'
    builder = Builder.for_qubits([a, b, m, m + 1])
    for _ in iter_chao_decompose_mpp4(
            a=None,
            b=None,
            c=a,
            d=b,
            m1=m,
            m2=m + 1,
            key=key,
            layer=5,
            basis='X',
            builder=builder):
        pass
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'X': [a]}, {'X': [a]}, []),
        ({'X': [b]}, {'X': [b]}, []),
        ({'Z': [a, b]}, {'Z': [a, b]}, []),
        ({'X': [a, b]}, {}, builder.tracker.current_measurement_record_targets_for([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


def test_chao_decompose_mzz2():
    a, b = 0, 1
    m = 1j
    key = 'key'
    builder = Builder.for_qubits([a, b, m, m + 1])
    for _ in iter_chao_decompose_mpp4(
            a=a,
            b=b,
            c=None,
            d=None,
            m1=m,
            m2=m + 1,
            key=key,
            layer=5,
            basis='Z',
            builder=builder):
        pass
    assert circuit_has_unsigned_stabilizers(builder.circuit, [
        ({'Z': [a]}, {'Z': [a]}, []),
        ({'Z': [b]}, {'Z': [b]}, []),
        ({'X': [a, b]}, {'X': [a, b]}, []),
        ({'Z': [a, b]}, {}, builder.tracker.current_measurement_record_targets_for([AtLayer(key, layer=5)])),
    ], q2i=builder.q2i)


@pytest.mark.parametrize('diameter,basis,rounds', itertools.product(
    range(3, 11),
    'XZ',
    [1, 2, 5, 10],
))
def test_code_distance(diameter: int, basis: str, rounds: int):
    task = chao_memory_experiment_task(basis=basis, rounds=rounds, diam=diameter, noise=0.001)
    min_err = task.circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
    assert len(min_err) == diameter
    if diameter <= 5 and rounds <= 5:
        min_err2 = task.circuit.search_for_undetectable_logical_errors(
            dont_explore_edges_increasing_symptom_degree=False,
            dont_explore_edges_with_degree_above=999,
            dont_explore_detection_event_sets_with_size_above=8,
            canonicalize_circuit_errors=True)
        assert len(min_err2) == diameter


def test_exact_surface_code_memory_circuit():
    task = chao_memory_experiment_task(basis='X', rounds=100, diam=3, noise=0.001)
    assert task.circuit == stim.Circuit("""
        QUBIT_COORDS(-2, 6) 0
        QUBIT_COORDS(-1, 6) 1
        QUBIT_COORDS(0, 0) 2
        QUBIT_COORDS(0, 4) 3
        QUBIT_COORDS(0, 8) 4
        QUBIT_COORDS(2, -2) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 6) 7
        QUBIT_COORDS(3, -2) 8
        QUBIT_COORDS(3, 2) 9
        QUBIT_COORDS(3, 6) 10
        QUBIT_COORDS(4, 0) 11
        QUBIT_COORDS(4, 4) 12
        QUBIT_COORDS(4, 8) 13
        QUBIT_COORDS(6, 2) 14
        QUBIT_COORDS(6, 6) 15
        QUBIT_COORDS(6, 10) 16
        QUBIT_COORDS(7, 2) 17
        QUBIT_COORDS(7, 6) 18
        QUBIT_COORDS(7, 10) 19
        QUBIT_COORDS(8, 0) 20
        QUBIT_COORDS(8, 4) 21
        QUBIT_COORDS(8, 8) 22
        QUBIT_COORDS(10, 2) 23
        QUBIT_COORDS(11, 2) 24
        RX 2 3 4 11 12 13 20 21 22
        R 0 6 15 23
        RX 5 7 14 16
        X_ERROR(0.001) 0 6 15 23
        Z_ERROR(0.001) 2 3 4 11 12 13 20 21 22 5 7 14 16
        DEPOLARIZE1(0.001) 1 8 9 10 17 18 19 24
        TICK
        MPP(0.001) X0 X2*X6 X12*X15 X20*X23
        CZ rec[-4] 0 rec[-3] 6 rec[-2] 15 rec[-1] 23
        RX 1 9 18 24
        MPP(0.001) Z5 Z3*Z7 Z11*Z14 Z13*Z16
        CX rec[-4] 5 rec[-3] 7 rec[-2] 14 rec[-1] 16
        R 8 10 17 19
        DEPOLARIZE1(0.001) 0 5
        DEPOLARIZE2(0.001) 2 6 12 15 20 23 3 7 11 14 13 16
        X_ERROR(0.001) 8 10 17 19
        Z_ERROR(0.001) 1 9 18 24
        DEPOLARIZE1(0.001) 4 21 22
        TICK
        MPP(0.001) Z0*Z1 Z6*Z9 Z15*Z18 Z23*Z24
        CX rec[-4] 0 rec[-3] 2 rec[-3] 6 rec[-2] 12 rec[-2] 15 rec[-1] 20 rec[-1] 23
        MPP(0.001) X5*X8 X7*X10 X14*X17 X16*X19
        CZ rec[-4] 5 rec[-3] 3 rec[-3] 7 rec[-2] 11 rec[-2] 14 rec[-1] 13 rec[-1] 16
        DEPOLARIZE2(0.001) 0 1 6 9 15 18 23 24 5 8 7 10 14 17 16 19
        DEPOLARIZE1(0.001) 2 3 4 11 12 13 20 21 22
        TICK
        MPP(0.001) X0 X3*X6 X13*X15 X21*X23
        CZ rec[-4] 0 rec[-4] 1 rec[-3] 6 rec[-3] 9 rec[-2] 15 rec[-2] 18 rec[-1] 23 rec[-1] 24
        MPP(0.001) Z5 Z7*Z12 Z14*Z20 Z16*Z22
        CX rec[-4] 5 rec[-4] 8 rec[-3] 7 rec[-3] 10 rec[-2] 14 rec[-2] 17 rec[-1] 16 rec[-1] 19
        DEPOLARIZE1(0.001) 0 5
        DEPOLARIZE2(0.001) 3 6 13 15 21 23 7 12 14 20 16 22
        DEPOLARIZE1(0.001) 1 2 4 8 9 10 11 17 18 19 24
        TICK
        M(0.001) 0 6 15 23
        CX rec[-3] 3 rec[-2] 13 rec[-1] 21
        MX(0.001) 5 7 14 16
        CZ rec[-3] 12 rec[-2] 20 rec[-1] 22
        DEPOLARIZE1(0.001) 0 6 15 23 5 7 14 16 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
        TICK
        R 0 6 15 23
        RX 5 7 14 16
        X_ERROR(0.001) 0 6 15 23
        Z_ERROR(0.001) 5 7 14 16
        DEPOLARIZE1(0.001) 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
        TICK
        MPP(0.001) X0*X3 X6*X11 X15*X21 X23
        CZ rec[-4] 0 rec[-3] 6 rec[-2] 15 rec[-1] 23
        MPP(0.001) Z2*Z5 Z4*Z7 Z12*Z14 Z16
        CX rec[-4] 5 rec[-3] 7 rec[-2] 14 rec[-1] 16
        DEPOLARIZE1(0.001) 23 16
        DEPOLARIZE2(0.001) 0 3 6 11 15 21 2 5 4 7 12 14
        DEPOLARIZE1(0.001) 1 8 9 10 13 17 18 19 20 22 24
        TICK
        MPP(0.001) Z0*Z1 Z6*Z9 Z15*Z18 Z23*Z24
        CX rec[-4] 0 rec[-4] 3 rec[-3] 6 rec[-3] 11 rec[-2] 15 rec[-2] 21 rec[-1] 23
        MPP(0.001) X5*X8 X7*X10 X14*X17 X16*X19
        CZ rec[-4] 2 rec[-4] 5 rec[-3] 4 rec[-3] 7 rec[-2] 12 rec[-2] 14 rec[-1] 16
        DEPOLARIZE2(0.001) 0 1 6 9 15 18 23 24 5 8 7 10 14 17 16 19
        DEPOLARIZE1(0.001) 2 3 4 11 12 13 20 21 22
        TICK
        MPP(0.001) X0*X4 X6*X12 X15*X22 X23
        MX(0.001) 1 9 18 24
        MPP(0.001) Z5*Z11 Z7*Z13 Z14*Z21 Z16
        M(0.001) 8 10 17 19
        DEPOLARIZE1(0.001) 23 1 9 18 24 16 8 10 17 19
        DEPOLARIZE2(0.001) 0 4 6 12 15 22 5 11 7 13 14 21
        DEPOLARIZE1(0.001) 2 3 20
        TICK
        M(0.001) 0 6 15 23
        CX rec[-4] 4 rec[-3] 12 rec[-2] 22
        MX(0.001) 5 7 14 16
        CZ rec[-4] 11 rec[-3] 13 rec[-2] 21
        DETECTOR(-2, 6, 0) rec[-24] rec[-20]
        DETECTOR(2, 2, 0) rec[-23] rec[-19]
        DETECTOR(6, 6, 0) rec[-22] rec[-18]
        DETECTOR(10, 2, 0) rec[-21] rec[-17]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 6 15 23 5 7 14 16 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
        TICK
        REPEAT 99 {
            R 0 6 15 23
            RX 5 7 14 16
            X_ERROR(0.001) 0 6 15 23
            Z_ERROR(0.001) 5 7 14 16
            DEPOLARIZE1(0.001) 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
            TICK
            MPP(0.001) X0 X2*X6 X12*X15 X20*X23
            CZ rec[-4] 0 rec[-3] 6 rec[-2] 15 rec[-1] 23
            RX 1 9 18 24
            MPP(0.001) Z5 Z3*Z7 Z11*Z14 Z13*Z16
            CX rec[-4] 5 rec[-3] 7 rec[-2] 14 rec[-1] 16
            R 8 10 17 19
            DEPOLARIZE1(0.001) 0 5
            DEPOLARIZE2(0.001) 2 6 12 15 20 23 3 7 11 14 13 16
            X_ERROR(0.001) 8 10 17 19
            Z_ERROR(0.001) 1 9 18 24
            DEPOLARIZE1(0.001) 4 21 22
            TICK
            MPP(0.001) Z0*Z1 Z6*Z9 Z15*Z18 Z23*Z24
            CX rec[-4] 0 rec[-3] 2 rec[-3] 6 rec[-2] 12 rec[-2] 15 rec[-1] 20 rec[-1] 23
            MPP(0.001) X5*X8 X7*X10 X14*X17 X16*X19
            CZ rec[-4] 5 rec[-3] 3 rec[-3] 7 rec[-2] 11 rec[-2] 14 rec[-1] 13 rec[-1] 16
            DEPOLARIZE2(0.001) 0 1 6 9 15 18 23 24 5 8 7 10 14 17 16 19
            DEPOLARIZE1(0.001) 2 3 4 11 12 13 20 21 22
            TICK
            MPP(0.001) X0 X3*X6 X13*X15 X21*X23
            CZ rec[-4] 0 rec[-4] 1 rec[-3] 6 rec[-3] 9 rec[-2] 15 rec[-2] 18 rec[-1] 23 rec[-1] 24
            MPP(0.001) Z5 Z7*Z12 Z14*Z20 Z16*Z22
            CX rec[-4] 5 rec[-4] 8 rec[-3] 7 rec[-3] 10 rec[-2] 14 rec[-2] 17 rec[-1] 16 rec[-1] 19
            DEPOLARIZE1(0.001) 0 5
            DEPOLARIZE2(0.001) 3 6 13 15 21 23 7 12 14 20 16 22
            DEPOLARIZE1(0.001) 1 2 4 8 9 10 11 17 18 19 24
            TICK
            M(0.001) 0 6 15 23
            CX rec[-3] 3 rec[-2] 13 rec[-1] 21
            MX(0.001) 5 7 14 16
            CZ rec[-3] 12 rec[-2] 20 rec[-1] 22
            DEPOLARIZE1(0.001) 0 6 15 23 5 7 14 16 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
            TICK
            R 0 6 15 23
            RX 5 7 14 16
            X_ERROR(0.001) 0 6 15 23
            Z_ERROR(0.001) 5 7 14 16
            DEPOLARIZE1(0.001) 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
            TICK
            MPP(0.001) X0*X3 X6*X11 X15*X21 X23
            CZ rec[-4] 0 rec[-3] 6 rec[-2] 15 rec[-1] 23
            MPP(0.001) Z2*Z5 Z4*Z7 Z12*Z14 Z16
            CX rec[-4] 5 rec[-3] 7 rec[-2] 14 rec[-1] 16
            DEPOLARIZE1(0.001) 23 16
            DEPOLARIZE2(0.001) 0 3 6 11 15 21 2 5 4 7 12 14
            DEPOLARIZE1(0.001) 1 8 9 10 13 17 18 19 20 22 24
            TICK
            MPP(0.001) Z0*Z1 Z6*Z9 Z15*Z18 Z23*Z24
            CX rec[-4] 0 rec[-4] 3 rec[-3] 6 rec[-3] 11 rec[-2] 15 rec[-2] 21 rec[-1] 23
            MPP(0.001) X5*X8 X7*X10 X14*X17 X16*X19
            CZ rec[-4] 2 rec[-4] 5 rec[-3] 4 rec[-3] 7 rec[-2] 12 rec[-2] 14 rec[-1] 16
            DEPOLARIZE2(0.001) 0 1 6 9 15 18 23 24 5 8 7 10 14 17 16 19
            DEPOLARIZE1(0.001) 2 3 4 11 12 13 20 21 22
            TICK
            MPP(0.001) X0*X4 X6*X12 X15*X22 X23
            MX(0.001) 1 9 18 24
            MPP(0.001) Z5*Z11 Z7*Z13 Z14*Z21 Z16
            M(0.001) 8 10 17 19
            DEPOLARIZE1(0.001) 23 1 9 18 24 16 8 10 17 19
            DEPOLARIZE2(0.001) 0 4 6 12 15 22 5 11 7 13 14 21
            DEPOLARIZE1(0.001) 2 3 20
            TICK
            M(0.001) 0 6 15 23
            CX rec[-4] 4 rec[-3] 12 rec[-2] 22
            MX(0.001) 5 7 14 16
            CZ rec[-4] 11 rec[-3] 13 rec[-2] 21
            DETECTOR(-2, 6, 0) rec[-96] rec[-92] rec[-24] rec[-20]
            DETECTOR(2, -2, 0) rec[-88] rec[-84] rec[-16] rec[-12]
            DETECTOR(2, 2, 0) rec[-95] rec[-91] rec[-23] rec[-19]
            DETECTOR(2, 6, 0) rec[-87] rec[-83] rec[-15] rec[-11]
            DETECTOR(6, 2, 0) rec[-86] rec[-82] rec[-14] rec[-10]
            DETECTOR(6, 6, 0) rec[-94] rec[-90] rec[-22] rec[-18]
            DETECTOR(6, 10, 0) rec[-85] rec[-81] rec[-13] rec[-9]
            DETECTOR(10, 2, 0) rec[-93] rec[-89] rec[-21] rec[-17]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 0 6 15 23 5 7 14 16 1 2 3 4 8 9 10 11 12 13 17 18 19 20 21 22 24
            TICK
        }
        MX(0.001) 2 3 4 11 12 13 20 21 22
        DETECTOR(-2, 6, 0) rec[-33] rec[-29] rec[-8] rec[-7]
        DETECTOR(2, 2, 0) rec[-32] rec[-28] rec[-9] rec[-8] rec[-6] rec[-5]
        DETECTOR(6, 6, 0) rec[-31] rec[-27] rec[-5] rec[-4] rec[-2] rec[-1]
        DETECTOR(10, 2, 0) rec[-30] rec[-26] rec[-3] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-9] rec[-6] rec[-3]
        DEPOLARIZE1(0.001) 2 3 4 11 12 13 20 21 22 0 1 5 6 7 8 9 10 14 15 16 17 18 19 23 24
    """)

