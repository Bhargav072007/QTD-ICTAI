"""Independent exhaustive oracle for bit order, pair convention and counters."""
import itertools
import numpy as np
from qiskit.quantum_info import Statevector
from phase2_qaoa.qubo_encoder import (build_qubo_matrix, qubo_to_hamiltonian,
    objective_value, encode_state_indices, decode_bitstring, PARAM_GRID)
from phase2_qaoa.qaoa_runner import QAOAExplorer


def test_all_basis_energies_and_roundtrips():
    explorer = QAOAExplorer()
    q = build_qubo_matrix()
    diagonal = qubo_to_hamiltonian(q).to_matrix().diagonal().real
    for indices in itertools.product(range(4), repeat=4):
        word = encode_state_indices(indices)
        assert decode_bitstring(word) == {name: PARAM_GRID[name][i] for name, i in zip(PARAM_GRID, indices)}
        x = np.array([int(c) for c in word[::-1]])
        # Independently express the pair-once polynomial, not matrix multiplication.
        expected = sum(q[i,i]*x[i] for i in range(8)) + sum(q[i,j]*x[i]*x[j] for i in range(8) for j in range(i+1,8))
        assert np.isclose(objective_value(q,x), expected, atol=1e-12)
        assert np.isclose(diagonal[int(word,2)], expected, atol=1e-12)
        assert np.isclose(explorer._hamiltonian_cost(word), expected, atol=1e-12)


def test_asymmetric_storage_and_constant_offset():
    q = np.array([[1.7, .2], [.6, -2.3]])
    h = qubo_to_hamiltonian(q)
    for n in range(4):
        x = np.array([n & 1, (n >> 1) & 1])
        expected = 1.7*x[0] - 2.3*x[1] + .4*x[0]*x[1]
        assert np.isclose(Statevector.from_int(n,4).expectation_value(h).real, expected)
        assert np.isclose(objective_value(q,x), expected)


def test_catalog_not_evaluated_and_resource_counts(monkeypatch):
    import phase2_qaoa.qaoa_runner as runner
    calls=[]
    original=runner.evaluate_state
    def tracked(state):
        calls.append(state)
        return original(state)
    monkeypatch.setattr(runner,'evaluate_state',tracked)
    explorer=runner.QAOAExplorer(shots=16)
    assert not calls
    result=explorer.run(k_iterations=1, output_path=None)
    resources=result['resources']
    assert resources['setup_catalog_calls'] == 0
    assert len(calls) == resources['simulator_calls_total_this_run']
    assert resources['optimizer_evaluations'] == sum(resources['optimizer_nfev_per_iteration'])
    assert resources['optimizer_shots'] == 16*resources['optimizer_evaluations']
    assert resources['sampling_shots'] == 16
    assert resources['circuit_shots_total'] == 16*(resources['optimizer_evaluations']+1)
