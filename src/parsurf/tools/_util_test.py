import stim

from parsurf.tools._util import circuit_has_unsigned_stabilizers


def test_circuit_has_unsigned_stabilizers():
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("H 0"),
        [
            ({"X": [0]}, {"Z": [0]}, []),
            ({"Z": [0]}, {"X": [0]}, []),
            ({"Y": [0]}, {"Y": [0]}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("CX 1 3"),
        [
            ({"X": [0]}, {"X": [0]}, []),
            ({"X": [1]}, {"X": [1, 3]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("CX 1 3"),
        [
            ({"X": [3]}, {"X": [1, 3]}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("RY 1"),
        [
            ({}, {"Y": [1]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("RX 1"),
        [
            ({}, {"Y": [1]}, []),
        ],
    )
    assert not circuit_has_unsigned_stabilizers(
        stim.Circuit("MY 3"),
        [
            ({"Y": [3]}, {}, []),
        ],
    )
    assert circuit_has_unsigned_stabilizers(
        stim.Circuit("MY 3"),
        [
            ({"Y": [3]}, {}, [stim.target_rec(-1)]),
        ],
    )
