import sinter

from parsurf.tools import NoiseModel


def generate_honeycomb_task(basis: str, rounds: int, diam: int, noise: float) -> sinter.Task:
    try:
        from hcb.codes.honeycomb.layout import HoneycombLayout
    except ImportError as ex:
        raise ImportError("Need to clone from https://github.com/Strilanc/honeycomb-boundaries and include its src directory in PYTHONPATH to generate honeycomb code circuits.") from ex

    assert diam % 2 == 1
    ideal_circuit = HoneycombLayout(
        data_width=diam + 1,
        data_height=(diam + 1) // 2 * 3,
        noise_level=noise,
        noisy_gate_set='EM3_v1',
        tested_observable='V' if basis == 'Z' else 'H',
        sheared=False,
        rounds=diam * 3 - 1,
    ).noisy_circuit().without_noise()
    noisy_circuit = NoiseModel.depolarizing_two_body_measurement_noise(noise).noisy_circuit(ideal_circuit)
    m = {
        'd': diam,
        'r': rounds,
        'b': basis,
        'p': noise,
        'c': 'honeycomb',
        'q': noisy_circuit.num_qubits,
    }

    return sinter.Task(
        circuit=noisy_circuit,
        json_metadata=m,
    )
