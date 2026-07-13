from phase2_qaoa.qubo_encoder import N_QUBITS, build_qubo_matrix, decode_bitstring, get_hamiltonian


def test_qubo_shape_and_qubits() -> None:
    qubo = build_qubo_matrix()
    hamiltonian, _ = get_hamiltonian()
    assert qubo.shape == (N_QUBITS, N_QUBITS)
    assert hamiltonian.num_qubits == N_QUBITS


def test_hamiltonian_is_hermitian() -> None:
    hamiltonian, _ = get_hamiltonian()
    if hasattr(hamiltonian, "is_hermitian"):
        assert hamiltonian.is_hermitian()
    else:
        assert all(abs(complex(coeff).imag) < 1e-12 for coeff in hamiltonian.coeffs)


def test_decode_bitstring_maps_to_valid_grid_values() -> None:
    decoded = decode_bitstring("01101001")
    assert decoded["int_heading"] in {75.0, 90.0, 105.0, 120.0}
    assert decoded["int_altitude"] in {28000.0, 30000.0, 31000.0, 32000.0}
