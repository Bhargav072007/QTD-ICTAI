from phase3_quantum_tree.classical_layer import run_classical_layer
from phase3_quantum_tree.pipeline import run_quantum_tree_pipeline


def test_classical_layer_learns_signal() -> None:
    result = run_classical_layer(seed=11)
    assert result["teacher_metrics"]["accuracy"] >= 0.7
    assert result["hidden"].shape[0] == len(result["states"])


def test_quantum_tree_pipeline_returns_summary_without_writing_outputs() -> None:
    result = run_quantum_tree_pipeline(seed=11, shots=64, output_path=None)
    assert len(result["architecture"]) == 3
    assert result["summary"]["student_mse"] >= 0.0
    assert len(result["top_distilled_states"]) >= 1
    assert len(result["cumulative_failures"]) == result["k_iterations"]
    assert result["total_unique_failures"] >= 0
    assert result["n_shots"] == 64


def test_quantum_tree_fast_mode_returns_small_budget_without_disk_dependency() -> None:
    result = run_quantum_tree_pipeline(seed=11, fast=True)
    assert result["fast_mode"] is True
    assert result["k_iterations"] == 5
    assert result["n_shots"] == 256
    assert len(result["cumulative_failures"]) == 5
